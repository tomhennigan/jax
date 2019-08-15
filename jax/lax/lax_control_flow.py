# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Control flow primitives.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import operator

import numpy as onp
import six

from jax import api
from jax import core
from jax.lax import lax
from jax.lax import _abstractify
from jax import linear_util as lu
from jax.abstract_arrays import ConcreteArray, ShapedArray, UnshapedArray
from jax.api_util import flatten_fun_nokwargs
from jax.interpreters import batching
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla
from jax.interpreters import ad
from jax.lib import xla_bridge as xb
from jax.util import partial, unzip2, safe_map, safe_zip, split_list
from jax.tree_util import tree_flatten, tree_unflatten, PyTreeDef, _get_node_type
from jax import ad_util

map = safe_map
zip = safe_zip
_reduce = six.moves.reduce


### fori_loop and while_loop

def fori_loop(lower, upper, body_fun, init_val):
  """Loop from ``lower`` to ``upper`` by reduction to ``while_loop``.

  The type signature in brief is

  .. code-block:: haskell

    fori_loop :: Int -> Int -> ((int, a) -> a) -> a -> a

  The semantics of ``fori_loop`` are given by this Python implementation::

    def fori_loop(lower, upper, body_fun, init_val):
      val = init_val
      for i in range(lower, upper):
        val = body_fun(i, val)
      return val

  Unlike that Python version, ``fori_loop`` is implemented in terms of a call to
  ``while_loop``. See the docstring for ``while_loop`` for more information.

  Args:
    lower: an integer representing the loop index lower bound (inclusive)
    upper: an integer representing the loop index upper bound (exclusive)
    body_fun: function of type ``(int, a) -> a``.
    init_val: initial loop carry value of type ``a``.

  Returns:
    Loop value from the final iteration, of type ``a``.
  """
  def while_cond_fun(loop_carry):
    i, _ = loop_carry
    return lax.lt(i, upper)

  def while_body_fun(loop_carry):
    i, x = loop_carry
    return lax.add(i, lax._const(i, 1)), body_fun(i, x)

  _, result = while_loop(while_cond_fun, while_body_fun, (lower, init_val))
  return result


def while_loop(cond_fun, body_fun, init_val):
  """Call ``body_fun`` repeatedly in a loop while ``cond_fun`` is True.

  The type signature in brief is

  .. code-block:: haskell

    while_loop :: (a -> Bool) -> (a -> a) -> a -> a

  The semantics of ``while_loop`` are given by this Python implementation::

    def while_loop(cond_fun, body_fun, init_val):
      val = init_val
      while cond_fun(val):
        val = body_fun(val)
      return val

  Unlike that Python version, ``while_loop`` is a JAX primitive and is lowered
  to a single XLA While HLO. That makes it useful for reducing compilation times
  for jit-compiled functions, since native Python loop constructs in an ``@jit``
  function are unrolled, leading to large XLA computations.

  Another difference from using Python-native loop constructs is that
  ``while_loop`` is not reverse-mode differentiable because XLA computations
  require static bounds on memory requirements.

  Args:
    cond_fun: function of type ``a -> Bool``.
    body_fun: function of type ``a -> a``.
    init_val: value of type ``a``, a type that can be a scalar, array, or any
      pytree (nested Python tuple/list/dict) thereof, representing the initial
      loop carry value.

  Returns:
    The output from the final iteration of body_fun, of type ``a``.
  """
  init_vals, in_tree = tree_flatten((init_val,))
  carry_pvals = map(_abstractify, init_vals)
  carry_avals, _ = unzip2(carry_pvals)

  cond_fun, _ = flatten_fun_nokwargs(lu.wrap_init(cond_fun), in_tree)
  cond_jaxpr, cond_pvals_out, cond_consts = pe.trace_to_jaxpr(cond_fun, carry_pvals)
  # TODO could check pytree out here
  (cond_out_pv, cond_out_const), = cond_pvals_out
  if cond_out_pv is None:
    # cond_fun evaluates to a constant, so don't need to generate a while_loop
    if cond_out_const:
      raise ValueError("infinite loop with no effects")
    else:
      return init_val
  else:
    if (not isinstance(cond_out_pv, ShapedArray) or cond_out_pv.shape
        or cond_out_pv.dtype != onp.bool_):
      msg = "while_loop cond_fun must return a scalar boolean, got {}."
      raise TypeError(msg.format(cond_out_pv))
    cond_out_avals = (cond_out_pv,)
  cond_const_avals, _ = unzip2(map(_abstractify, cond_consts))
  cond_jaxpr = core.TypedJaxpr(pe.closure_convert_jaxpr(cond_jaxpr),
                               (), carry_avals + cond_const_avals, cond_out_avals)
  cond_nconsts = len(cond_consts)

  body_fun, out_tree = flatten_fun_nokwargs(lu.wrap_init(body_fun), in_tree)
  body_jaxpr, body_pvals_out, body_consts = pe.trace_to_jaxpr(
      body_fun, carry_pvals, instantiate=True)
  carry_avals_out, _ = unzip2(body_pvals_out)
  if (out_tree(),) != in_tree.children:
    raise TypeError("body_fun input and output must have identical structure, "
                    "got input {} and output {}.".format(in_tree, out_tree()))
  if list(carry_avals) != map(core.lattice_join, carry_avals, carry_avals_out):
    msg = ("while_loop body_fun output type does not match its input type: "
           "input is {} and output is {}.")
    raise TypeError(msg.format(carry_avals, carry_avals_out))
  body_const_avals, _ = unzip2(map(_abstractify, body_consts))
  body_jaxpr = core.TypedJaxpr(pe.closure_convert_jaxpr(body_jaxpr),
                               (), carry_avals + body_const_avals, carry_avals_out)
  body_nconsts = len(body_consts)

  outs = while_p.bind(
      *itertools.chain(cond_consts, body_consts, init_vals),
      cond_nconsts=cond_nconsts, cond_jaxpr=cond_jaxpr,
      body_nconsts=body_nconsts, body_jaxpr=body_jaxpr)
  return tree_unflatten(out_tree(), outs)

def _while_loop_impl(*args, **kwargs):
  # TODO replace this with a call to apply_primitive
  cond_jaxpr, cond_nconsts = kwargs.pop("cond_jaxpr"), kwargs.pop("cond_nconsts")
  body_jaxpr, body_nconsts = kwargs.pop("body_jaxpr"), kwargs.pop("body_nconsts")
  assert not kwargs
  cond_consts, body_consts, init_vals = split_list(args, [cond_nconsts, body_nconsts])

  cond_fun = partial(core.eval_jaxpr, cond_jaxpr.jaxpr, (), ())
  body_fun = partial(core.eval_jaxpr, body_jaxpr.jaxpr, (), ())

  vals = init_vals
  while cond_fun(*(cond_consts + vals))[0]:
    vals = body_fun(*(body_consts + vals))
  return vals

def _while_loop_abstract_eval(*args, **kwargs):
  return kwargs["body_jaxpr"].out_avals

def _while_loop_translation_rule(c, axis_env, *args, **kwargs):
  cond_jaxpr, cond_nconsts = kwargs.pop("cond_jaxpr"), kwargs.pop("cond_nconsts")
  body_jaxpr, body_nconsts = kwargs.pop("body_jaxpr"), kwargs.pop("body_nconsts")
  assert not kwargs
  cond_consts, body_consts, init_vals = split_list(args, [cond_nconsts, body_nconsts])

  # Since jaxprs don't have tuples and have multiple return values, but we need
  # the HLO While loop to take a single tuple input and output a single boolean
  # (for the cond computation) or a single tuple output (for the body
  # computation), we build XLA computations that handle the tuple munging before
  # generating a Call into the computations formed from the jaxprs.

  loop_carry = c.Tuple(*(cond_consts + body_consts + init_vals))
  carry_shape = c.GetShape(loop_carry)

  cond_c = xb.make_computation_builder("cond_computation")
  cond_carry = cond_c.ParameterWithShape(carry_shape)
  cond_carry_elts = [cond_c.GetTupleElement(cond_carry, i) for i in range(len(args))]
  x, _, z = split_list(cond_carry_elts, [cond_nconsts, body_nconsts])
  cond_outs = cond_c.Call(
      xla.jaxpr_computation(cond_jaxpr.jaxpr, axis_env, (), (),
                            *map(cond_c.GetShape, x + z)), x + z)
  cond_c = cond_c.Build(cond_c.GetTupleElement(cond_outs, 0))

  body_c = xb.make_computation_builder("body_computation")
  body_carry = body_c.ParameterWithShape(carry_shape)
  body_carry_elts = [body_c.GetTupleElement(body_carry, i) for i in range(len(args))]
  x, y, z = split_list(cond_carry_elts, [cond_nconsts, body_nconsts])
  body_out = body_c.Call(
      xla.jaxpr_computation(body_jaxpr.jaxpr, axis_env, (), (),
                            *map(body_c.GetShape, y + z)), y + z)
  z = [body_c.GetTupleElement(body_out, i) for i in range(len(init_vals))]
  body_c = body_c.Build(body_c.Tuple(*(x + y + z)))

  ans = c.While(cond_c, body_c, loop_carry)
  ans_elts = [c.GetTupleElement(ans, i) for i in range(len(args))]
  _,  _, z = split_list(ans_elts, [cond_nconsts, body_nconsts])
  return c.Tuple(*z)

def _while_loop_batching_rule(args, dims, avals_out, cond_jaxpr, body_jaxpr):
  # See https://github.com/google/jax/issues/441 for a discussion.
  # To batch a while_loop, we need to do some masking, since the elements of the
  # batch may run for different numbers of iterations. We perform that masking
  # using lax.select, and keep the loop running so long as any of the batch
  # elements need by effectively using an np.any(...) in the cond_fun.
  # The basic strategy here is to lift `cond_jaxpr` and `body_jaxpr` back into
  # traceable Python functions using `core.eval_jaxpr`. Then we can batch them
  # using `batching.batch_transform` (the transform underlying `api.vmap`).

  sz, = {x.shape[d] for x, d in zip(args, dims) if d is not batching.not_mapped}
  args = [x if d is batching.not_mapped else moveaxis(x, d, 0)
          for x, d in zip(args, dims)]

  init_val, cond_consts, body_consts = \
      _unpack_while_loop_args(batched_args, cond_jaxpr, body_jaxpr)
  init_val_bd, cond_consts_bd, body_consts_bd = \
      _unpack_while_loop_args(batch_dims, cond_jaxpr, body_jaxpr)

  # TODO
  import ipdb; ipdb.set_trace()
  # for _ in range(1000):
  #   batched_in = [d is not batching.not_mapped for d in 
  #   body_jaxpr_batched, batched_out = batching.batch_jaxpr(body_jaxpr, sz, 

  ###

  # TODO(mattjj): if cond_consts_bd is also None, we could keep cond_fun
  # unbatched and avoid the masking logic, but we ignore that optimization
  init_val = batching.bdim_at_front(init_val, init_val_bd, size,
                                    force_broadcast=True)
  init_val_bd = 0

  def batched_cond_fun(batched_loop_carry):
    @lu.wrap_init
    def lifted(loop_carry, cond_consts):
      return core.eval_jaxpr(cond_jaxpr, cond_consts, (), loop_carry)
    f = batching.batch_transform(lifted, size, (init_val_bd, cond_consts_bd), 0)
    preds = f.call_wrapped((batched_loop_carry, cond_consts))
    return lax.reduce(preds, onp.array(False), lax.bitwise_or, [0])

  def batched_body_fun(batched_loop_carry):
    @lu.wrap_init
    def lifted(loop_carry, cond_consts, body_consts):
      pred = core.eval_jaxpr(cond_jaxpr, cond_consts, (), loop_carry)
      new_loop_carry = core.eval_jaxpr(body_jaxpr, body_consts, (), loop_carry)
      return _jaxtupletree_select(pred, new_loop_carry, loop_carry)
    f = batching.batch_transform(
        lifted, size, (init_val_bd, cond_consts_bd, body_consts_bd), init_val_bd)
    return f.call_wrapped((batched_loop_carry, cond_consts, body_consts))

  return while_loop(batched_cond_fun, batched_body_fun, init_val), init_val_bd

def _jaxtupletree_select(pred, on_true, on_false):
  aval = core.get_aval(on_true)
  if type(aval) is core.AbstractTuple:
    return core.pack(map(partial(_jaxtupletree_select, pred), on_true, on_false))
  elif isinstance(aval, UnshapedArray):
    return lax.select(pred, on_true, on_false)
  else:
    raise TypeError(aval)


while_p = lax.Primitive('while')
while_p.multiple_results = True
while_p.def_impl(_while_loop_impl)
while_p.def_abstract_eval(_while_loop_abstract_eval)
xla.initial_style_translations[while_p] = _while_loop_translation_rule
batching.primitive_batchers[while_p] = _while_loop_batching_rule


### cond

def cond(pred, true_operand, true_fun, false_operand, false_fun):
  def trace_jaxpr(fun, operand):
    ops_flat, in_tree = tree_flatten((operand,))
    in_avals, _ = map(_abstractify, ops_flat)
    fun_flat, out_tree = flatten_fun_nokwargs(lu.wrap_init(fun), in_tree)
    jaxpr = _make_typed_jaxpr(fun_flat, in_avals)
    return ops_flat, jaxpr, out_tree()

  true_ops, true_jaxpr, true_tree = trace_jaxpr(true_fun, true_operand)
  false_ops, false_jaxpr, false_tree = trace_jaxpr(false_fun, false_operand)

  if true_tree != false_tree:
    msg = "true_fun and false_fun outputs must have identical structure"
    raise TypeError(msg)

  try:
    joined_pvals = map(pe.join_pvals, true_pvals, false_pvals)
  except TypeError:
    msg = ("could not merge true_fun and false_fun outputs to consistent type: "
           "{} and {}".format(true_pvals, false_pvals))
    raise TypeError(msg.format(true_pvals, false_pvals))
  true_jaxpr, true_consts = _revise_cond_jaxpr(joined_pvals, true_pvals,
                                               true_jaxpr, true_consts)
  false_jaxpr, false_consts = _revise_cond_jaxpr(joined_pvals, false_pvals,
                                                 false_jaxpr, false_consts)

  out = cond_p.bind(*itertools.chain([pred], true_op, true_consts,
                                     false_op, false_consts),
                    true_jaxpr=true_jaxpr, false_jaxpr=false_jaxpr)
  out = pe.merge_pvals(out, joined_pval)
  return tree_unflatten(true_tree, out)

def _revise_cond_jaxpr(new_pval, old_pval, jaxpr, consts):
  new_pv, new_const = new_pval
  old_pv, old_const = old_pval
  if new_pv == old_pv:
    # we didn't move up the lattice by joining with the other side
    return jaxpr, consts
  elif old_pv is None:
    # we moved up the lattice from totally-known, so make a new jaxpr that
    # returns a single constant JaxTuple with elements that are constants
    # drawn from consts where new_pv is unknown
    assert not jaxpr.eqns and not consts
    outvar = pe.Var(0, "_cond")
    new_jaxpr = jaxpr.copy()
    new_jaxpr.constvars = [outvar]
    new_jaxpr.outvar = outvar
    new_consts = (core.pack([core.unit if pv is None else old_c
                             for pv, old_c in zip(new_pv, old_const)]),)
    return new_jaxpr, new_consts
  else:
    # we moved up the lattice, but not from totally-constant, so adapt the
    # japxr to return some new constants in places that are now unknown but
    # weren't before
    eqn = jaxpr.eqns[-1]
    assert eqn.primitive == core.pack_p
    assert len(eqn.outvars) == 1 and eqn.outvars[0] == jaxpr.outvar
    newvar = pe.gensym("_cond")
    new_constvars, new_constvals = unzip2(
        [(newvar(), c) for new, old, c in zip(new_pv, old_pv, old_const)
         if old is None and new is not None])
    new_consts = consts + tuple(new_constvals)
    new_jaxpr = jaxpr.copy()
    new_jaxpr.constvars = tuple(jaxpr.constvars) + tuple(new_constvars)
    newvars = iter(new_constvars)
    new_invars = [next(newvars) if old is None and new is not None else
                  (core.unitvar if new is None and old is None else v)
                  for new, old, v in zip(new_pv, old_pv, eqn.invars)]
    new_jaxpr.eqns = (list(jaxpr.eqns[:-1]) +
                      [_pack_eqn(new_invars, jaxpr.outvar)])
    return new_jaxpr, new_consts

def _unpack_eqn(invar, outvars):
  return core.JaxprEqn([invar], outvars, core.identity_p, (), False, True, {})

def _pack_eqn(invars, outvar):
  return core.JaxprEqn(invars, [outvar], core.pack_p, (), False, False, {})


def _cond_impl(pred, true_op, true_consts, false_op, false_consts, aval_out,
               true_jaxpr, false_jaxpr):
  true_fun = partial(core.eval_jaxpr, true_jaxpr, true_consts, ())
  false_fun = partial(core.eval_jaxpr, false_jaxpr, false_consts, ())

  if pred:
    return true_fun(true_op)
  else:
    return false_fun(false_op)

def _cond_abstract_eval(pred, true_op, true_consts, false_op, false_consts,
                        aval_out, true_jaxpr, false_jaxpr):
  if not isinstance(pred, ShapedArray) or pred.shape or pred.dtype != onp.bool_:
    msg = "cond pred must be a scalar boolean type, got {}."
    raise TypeError(msg.format(pred))
  if isinstance(pred, ConcreteArray):
    raise NotImplementedError  # TODO(mattjj)
  else:
    return _maybe_tracer_tuple_to_abstract_tuple(aval_out)


def _cond_translation_rule(c, axis_env, pred, true_op, true_consts, false_op,
                           false_consts, aval_out, true_jaxpr, false_jaxpr):
  def make_computation(jaxpr, operand):
    assert len(jaxpr.invars) == 1
    arg_var = pe.Var(0, "arg")
    consts_var = pe.Var(0, "consts")
    jaxpr_converted = jaxpr.copy()
    jaxpr_converted.constvars = []
    jaxpr_converted.invars = [arg_var]
    jaxpr_converted.eqns = (
        [_unpack_eqn(arg_var, [jaxpr.invars[0], consts_var]),
        _unpack_eqn(consts_var, jaxpr.constvars)]
        + list(jaxpr.eqns))
    return xla.jaxpr_computation(jaxpr_converted, axis_env, (), (),
                                 c.GetShape(operand))

  true_arg = c.Tuple(true_op, true_consts)
  true_comp = make_computation(true_jaxpr, true_arg)

  false_arg = c.Tuple(false_op, false_consts)
  false_comp = make_computation(false_jaxpr, false_arg)

  return c.Conditional(pred, true_arg, true_comp, false_arg, false_comp)

cond_p = lax.Primitive('cond')
cond_p.def_impl(_cond_impl)
cond_p.def_abstract_eval(_cond_abstract_eval)
xla.initial_style_translations[cond_p] = _cond_translation_rule


def _maybe_tracer_tuple_to_abstract_tuple(tup):
  if isinstance(tup, pe.JaxprTracerTuple):
    return core.AbstractTuple(list(map(_maybe_tracer_tuple_to_abstract_tuple, tup)))
  elif isinstance(tup, core.AbstractValue):
    return tup
  elif tup is None:
    return core.AbstractTuple(())
  else:
    raise TypeError(tup)


### scan

def _convert_zeros(instantiate, example, tangent):
  assert False, "update it"
  t = type(instantiate)
  if t is bool:
    if instantiate:
      return ad.instantiate_zeros(example, tangent)
    elif tangent is ad_util.zero:
      return core.unit
    else:
      raise TypeError(tangent)  # not clear if ever reachable
  elif t is tuple:
    if type(tangent) is ad.TangentTuple:
      return core.pack(map(_convert_zeros, instantiate, example, tangent))
    elif tangent is ad_util.zero:
      zeros = [ad_util.zero] * len(instantiate)
      return core.pack(map(_convert_zeros, instantiate, example, zeros))
    else:
      raise TypeError(tangent)
  else:
    raise TypeError(t)

class FixedPointError(Exception): pass


def scan(f, init, xs):
  """Scan a function over leading array axes while carrying along state.

  The type signature in brief is

  .. code-block:: haskell

    scan :: (c -> a -> (c, b)) -> c -> [a] -> (c, [b])

  where we use [t] here to denote the type t with an additional leading axis.
  That is, if t is an array type then [t] represents the type with an additional
  leading axis, and if t is a pytree (container) type with array leaves then [t]
  represents the type with the same pytree structure and corresponding leaves
  each with an additional leading axis.

  When both ``a`` and ``b`` are array types, the semantics of ``scan`` are given
  by this Python implementation::

    def scan(f, init, xs):
      carry = init
      ys = []
      for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
      return carry, np.stack(ys)

  Unlike that Python version, both ``a`` and ``b`` may be arbitrary pytree
  types, and so multiple arrays can be scanned over at once and produce multiple
  output arrays.

  Also unlike that Python version, ``scan`` is a JAX primitive and is lowered to
  a single XLA While HLO. That makes it useful for reducing compilation times
  for jit-compiled functions, since native Python loop constructs in an ``@jit``
  function are unrolled, leading to large XLA computations.

  Args:
    f: a Python function to be scanned of type ``c -> a -> (c, b)``, meaning
      that ``f`` accepts two arguments where the first is a value of the loop
      carry and the second is a slice of ``xs`` along its leading axis, and that
      ``f`` returns a pair where the first element represents a new value for
      the loop carry and the second represents a slice of the output.
    init: an initial loop carry value of type ``c``, which can be a scalar,
      array, or any pytree (nested Python tuple/list/dict) thereof, representing
      the initial loop carry value.
    xs: the value of type ``[a]`` over which to scan along the leading axis,
      where ``[a]`` can be an array or any pytree (nested Python
      tuple/list/dict) thereof with consistent leading axis sizes.

  Returns:
    A pair of type ``(c, [b])`` where the first element represents the final
    loop carry value and the second element represents the stacked outputs of
    the second output of ``f`` when scanned over the leading axis of the inputs.
  """
  num_carry = len(tree_flatten(init)[0])
  in_flat, in_tree = tree_flatten((init, xs))
  init_flat, xs_flat = in_flat[:num_carry], in_flat[num_carry:]
  f, out_tree = flatten_fun_nokwargs(lu.wrap_init(f), in_tree)
  length, = {x.shape[0] for x in xs_flat}  # TODO value error

  carry_pvals = map(_abstractify, init_flat)
  carry_avals, _ = unzip2(carry_pvals)
  x_pvals = [pe.PartialVal((ShapedArray(aval.shape[1:], aval.dtype), core.unit))
             for aval, _ in map(_abstractify, xs_flat)]
  x_avals, _ = unzip2(x_pvals)
  jaxpr, pvals_out, consts = pe.trace_to_jaxpr(f, carry_pvals + x_pvals,
                                               instantiate=True)
  carry_pvals_out, y_pvals = pvals_out[:num_carry], pvals_out[num_carry:]
  carry_avals_out, _ = unzip2(carry_pvals_out)
  assert carry_avals_out == carry_avals  # TODO type error
  y_avals, _ = unzip2(y_pvals)
  const_avals, _ = unzip2(map(_abstractify, consts))
  jaxpr = core.TypedJaxpr(pe.closure_convert_jaxpr(jaxpr),
                          (), const_avals + carry_avals + x_avals,
                          carry_avals + y_avals)
  out = scan_p.bind(*itertools.chain(consts, in_flat),
                    forward=True, length=length, jaxpr=jaxpr,
                    num_consts=len(consts), num_carry=num_carry,
                    linear=(False,) * (len(consts) + len(in_flat)))
  return tree_unflatten(out_tree(), out)

def _scan_impl(*args, **kwargs):
  forward, length = kwargs.pop("forward"), kwargs.pop("length")
  num_consts, num_carry = kwargs.pop("num_consts"), kwargs.pop("num_carry")
  jaxpr, _ = kwargs.pop("jaxpr"), kwargs.pop("linear")
  assert not kwargs
  consts, init, xs = split_list(args, [num_consts, num_carry])
  _, _, x_avals = split_list(jaxpr.in_avals, [num_consts, num_carry])
  _, y_avals = split_list(jaxpr.out_avals, [num_carry])

  def body_fun(i, vals):
    i = i if forward else length - i - 1
    carry, ys = split_list(vals, [num_carry])
    x = map(partial(_index_array, i), x_avals, xs)
    out_flat = core.jaxpr_as_fun(jaxpr)(*(consts + carry + x))
    carry_out, y_updates = split_list(out_flat, [num_carry])
    ys_out = map(partial(_update_array, i), y_avals, ys, y_updates)
    return carry_out + ys_out

  ys_init = map(partial(_empty_array, length), y_avals)
  return fori_loop(0, length, body_fun, init + ys_init)

def _index_array(i, aval, x):
  if aval is core.abstract_unit:
    return core.unit
  else:
    return lax.dynamic_index_in_dim(x, i, keepdims=False)

def _empty_array(sz, aval):
  if aval is core.abstract_unit:
    return core.unit
  else:
    return lax.full((sz,) + aval.shape, 0, aval.dtype)

def _update_array(i, aval, xs, x):
  if aval is core.abstract_unit:
    return core.unit
  else:
    return lax.dynamic_update_index_in_dim(xs, x, i, 0)

def _scan_jvp(primals, tangents, forward, length, jaxpr, num_consts, num_carry,
              linear):
  num_xs = len(jaxpr.in_avals) - num_carry - num_consts
  num_ys = len(jaxpr.out_avals) - num_carry
  nonzeros = [t is not ad_util.zero for t in tangents]
  const_nz, init_nz, xs_nz = split_list(nonzeros, [num_consts, num_carry])

  carry_nz = init_nz
  for _ in range(1000):
    nonzeros = const_nz + carry_nz + xs_nz
    jaxpr_jvp, nonzeros_out = ad.jvp_jaxpr(
        jaxpr, nonzeros, instantiate=carry_nz + [False] * num_ys)
    carry_nz_out, ys_nz = nonzeros_out[:num_carry], nonzeros_out[num_carry:]
    if carry_nz_out == carry_nz:
      break
    else:
      carry_nz = carry_nz_out
  else:
    raise FixedPointError
  tangents = [ad.instantiate_zeros(x, t) if t is ad_util.zero and nz else t
              for x, t, nz in zip(primals, tangents, nonzeros)]

  consts, init, xs = split_list(primals, [num_consts, num_carry])
  all_tangents = split_list(tangents, [num_consts, num_carry])
  consts_dot, init_dot, xs_dot = map(_prune_zeros, all_tangents)

  jaxpr_jvp_rearranged = ad.rearrange_binders(
      jaxpr_jvp,
      [num_consts, num_carry, num_xs], [len(consts_dot), len(init_dot), len(xs_dot)],
      [num_carry, num_ys], [len(init_dot), sum(nonzeros_out) - len(init_dot)])

  consts_linear, init_linear, xs_linear = split_list(linear, [num_consts, num_carry])
  jaxpr_jvp_linear = (consts_linear + [True] * len(consts_dot)
                      + init_linear + [True] * len(init_dot)
                      + xs_linear + [True] * len(xs_dot))

  out_flat = scan_p.bind(
      *(consts + consts_dot + init + init_dot + xs + xs_dot),
      forward=forward, length=length, jaxpr=jaxpr_jvp_rearranged,
      num_consts=num_consts+len(consts_dot), num_carry=num_carry+len(init_dot),
      linear=jaxpr_jvp_linear)

  carry, carry_dot, ys, ys_dot = split_list(out_flat, [num_carry, len(init_dot), num_ys])
  primals_out = carry + ys
  tangents_out = iter(carry_dot + ys_dot)
  tangents_out = [next(tangents_out) if nz else ad_util.zero for nz in nonzeros_out]
  return primals_out, tangents_out

def _prune_zeros(ts):
  return [t for t in ts if t is not ad_util.zero]

def _scan_partial_eval(trace, *tracers, **kwargs):
  forward, length = kwargs.pop("forward"), kwargs.pop("length")
  num_consts, num_carry = kwargs.pop("num_consts"), kwargs.pop("num_carry")
  jaxpr, linear = kwargs.pop("jaxpr"), kwargs.pop("linear")
  assert not kwargs
  num_xs = len(jaxpr.in_avals) - num_carry - num_consts
  num_ys = len(jaxpr.out_avals) - num_carry

  unknowns = original_unknowns = [t.pval[0] is not None for t in tracers]
  const_uk, init_uk, xs_uk = split_list(unknowns, [num_consts, num_carry])

  carry_uk = init_uk
  for _ in range(1000):
    unknowns = const_uk + carry_uk + xs_uk
    jaxpr_1, jaxpr_2, out_uk = pe.partial_eval_jaxpr(
        jaxpr, unknowns, instantiate=carry_uk + [False] * num_ys)
    carry_uk_out, ys_uk = out_uk[:num_carry], out_uk[num_carry:]
    if carry_uk_out == carry_uk:
      break
    else:
      carry_uk = carry_uk_out
  else:
    raise FixedPointError

  in_consts = [core.unit if uk else t.pval[1] for uk, t in zip(unknowns, tracers)]
  new_tracers = [trace.instantiate_const(t) if uk else trace.new_instantiated_literal(core.unit)
                 for uk, t in zip(unknowns, tracers)]

  carry_avals, y_avals = split_list(jaxpr.out_avals, [num_carry])
  ys_avals = map(partial(_promote_aval_rank, length), y_avals)
  out_avals = carry_avals + ys_avals
  out_pvs = [aval if uk else None for aval, uk in zip(out_avals, out_uk)]

  linear_1 = [lin or uk for lin in linear]
  out_flat = scan_p.bind(
      *in_consts, forward=forward, length=length, jaxpr=jaxpr_1,
      num_consts=num_consts, num_carry=num_carry, linear=linear_1)
  out_carry, ys, residuals = split_list(out_flat, [num_carry, num_ys])
  out_consts = out_carry + ys
  residual_tracers = map(trace.new_instantiated_const, residuals)
  out_tracers = [pe.JaxprTracer(trace, pe.PartialVal((pv, const)), None)
                 for pv, const in zip(out_pvs, out_consts)]
  linear_2 = ([lin or not uk for uk, lin in zip(unknowns, linear)]
              + [False] * len(residual_tracers))
  eqn = pe.new_jaxpr_eqn(new_tracers + residual_tracers, out_tracers, scan_p,
                         (), dict(forward=forward, length=length, jaxpr=jaxpr_2,
                                  num_consts=num_consts, num_carry=num_carry,
                                  linear=linear_2))
  for t in out_tracers: t.recipe = eqn
  return out_tracers

def _promote_aval_rank(sz, aval):
  if aval is core.abstract_unit:
    return core.abstract_unit
  else:
    return ShapedArray((sz,) + aval.shape, aval.dtype)

def _scan_transpose(cts, *args, **kwargs):
  forward, length = kwargs.pop("forward"), kwargs.pop("length")
  num_consts, num_carry = kwargs.pop("num_consts"), kwargs.pop("num_carry")
  jaxpr, linear = kwargs.pop("jaxpr"), kwargs.pop("linear")
  assert not kwargs
  num_ys = len(jaxpr.out_avals) - num_carry

  # we can only transpose scans for which the nonlinear values appear in xs
  consts_lin, init_lin, xs_lin = split_list(linear, [num_consts, num_carry])
  num_lin = sum(xs_lin)
  if not all(consts_lin) or not all(init_lin) or not all(xs_lin[:num_lin]):
    raise NotImplementedError

  consts, init, xs, res = split_list(args, [num_consts, num_carry, num_lin])
  assert not any(r is ad.undefined_primal for r in res)

  carry_avals, y_avals = split_list(jaxpr.out_avals, [num_carry])
  ys_avals = map(partial(_promote_aval_rank, length), y_avals)
  ct_carry, ct_ys = split_list(cts, [num_carry])
  ct_carry = map(ad.instantiate_zeros_aval, carry_avals, ct_carry)
  ct_ys = map(ad.instantiate_zeros_aval, ys_avals, ct_ys)
  ct_consts = map(ad_util.zeros_like_aval, jaxpr.in_avals[:num_consts])

  #       jaxpr :: [T d] -> [T c] -> [T a, res] -> ([T c], [T b])
  # jaxpr_trans :: [] -> [CT d, CT c] -> [CT b, res] -> ([CT d, CT c], [CT a])
  jaxpr_trans = _transpose_jaxpr2(num_consts, len(res), jaxpr)
  linear_trans = ([True] * (len(ct_consts) + len(ct_carry) + len(ct_ys))
                  + [False] * len(res))

  outs = scan_p.bind(
      *(ct_consts + ct_carry + ct_ys + res), forward=not forward, length=length,
      jaxpr=jaxpr_trans, num_consts=0, num_carry=num_consts+num_carry,
      linear=linear_trans)
  ct_consts, ct_init, ct_xs = split_list(outs, [num_consts, num_carry])
  return ct_consts + ct_init + ct_xs + [None] * len(res)

# # transpose_jaxpr :: Int -> ([a, res] -> b) -> ([CT b, res] -> CT a)
# def _transpose_jaxpr(num_res, jaxpr):
#   num_lin = len(jaxpr.in_avals) - num_res
#   num_out = len(jaxpr.out_avals)

#   @lu.wrap_init
#   def transposed(*bbar_res):
#     b_bar, res = bbar_res[:num_out], bbar_res[num_out:]
#     primals = (ad.undefined_primal,) * num_lin + res
#     _, abar_res = ad.backward_pass(jaxpr.jaxpr, jaxpr.literals, (), primals, b_bar)
#     return map(ad.instantiate_zeros_aval, jaxpr.in_avals[:num_lin], abar_res[:num_lin])
#   return _make_typed_jaxpr(transposed, jaxpr.out_avals + jaxpr.in_avals[num_lin:])

# transpose_jaxpr2 :: ([c, a, res] -> b) -> ([CT c, CT b, res] -> [CT c, CT a]
def _transpose_jaxpr2(num_c, num_res, jaxpr):
  num_a = len(jaxpr.in_avals) - num_c - num_res
  c_avals, a_avals, res_avals = split_list(jaxpr.in_avals, [num_c, num_a])
  num_b = len(jaxpr.out_avals)
  b_avals = list(jaxpr.out_avals)

  @lu.wrap_init
  def transposed(*cbar_bbar_res):
    c_bar, b_bar, res = split_list(cbar_bbar_res, [num_c, num_b])
    primals = [ad.undefined_primal] * (num_c + num_a) + res
    _, cbar_abar = ad.backward_pass(jaxpr.jaxpr, jaxpr.literals, (), primals,
                                    b_bar)
    new_c_bar, a_bar, _ = split_list(cbar_abar, [num_c, num_a])
    a_bar = map(ad.instantiate_zeros_aval, a_avals, a_bar)
    c_bar = map(ad.instantiate_zeros_aval, c_avals,
                map(ad.add_tangents, c_bar, new_c_bar))
    return c_bar + a_bar
  return _make_typed_jaxpr(transposed, c_avals + b_avals + res_avals)

def _make_typed_jaxpr(traceable, in_avals):
  pvals = [pe.PartialVal((aval, core.unit)) for aval in in_avals]
  jaxpr, pvals_out, consts = pe.trace_to_jaxpr(traceable, pvals, instantiate=True)
  out_avals, _ = unzip2(pvals_out)
  return core.TypedJaxpr(jaxpr, consts, in_avals, out_avals)


def _scan_batching_rule(args, dims, forward, length, jaxpr, num_consts,
                        num_carry, linear):
  num_ys = len(jaxpr.out_avals) - num_carry
  size, = {x.shape[d] for x, d in zip(args, dims) if d is not batching.not_mapped}
  orig_batched = [d is not batching.not_mapped for d in dims]
  const_batched, init_batched, xs_batched = split_list(orig_batched, [num_consts, num_carry])

  carry_batched = init_batched
  for _ in range(1000):
    batched = const_batched + carry_batched + xs_batched
    jaxpr_batched, batched_out = batching.batch_jaxpr(
        jaxpr, size, batched, instantiate=carry_batched + [False] * num_ys)
    carry_batched_out, ys_batched = batched_out[:num_carry], batched_out[num_carry:]
    if carry_batched_out == carry_batched:
      break
    else:
      carry_batched = carry_batched_out
  else:
    raise FixedPointError

  consts, init, xs = split_list(args, [num_consts, num_carry])
  consts_bdims, init_bdims, xs_bdims = split_list(dims, [num_consts, num_carry])
  new_consts = [batching.moveaxis(x, d, 0) if d is not batching.not_mapped and d != 0
                else x for x, d in zip(consts, consts_bdims)]
  new_init = [batching.broadcast(x, size, 0) if now_batched and not was_batched
              else batching.moveaxis(x, d, 0) if now_batched else x
              for x, d, was_batched, now_batched in
              zip(init, init_bdims, init_batched, carry_batched)]
  new_xs = [batching.moveaxis(x, d, 1) if d is not batching.not_mapped and d != 1
            else x for x, d in zip(xs, xs_bdims)]
  new_args = new_consts + new_init + new_xs

  outs = scan_p.bind(*new_args, forward=forward, length=length, jaxpr=jaxpr_batched,
                     num_consts=num_consts, num_carry=num_carry, linear=linear)
  carry_bdims = [0 if b else batching.not_mapped for b in carry_batched]
  ys_bdims = [1 if b else batching.not_mapped for b in ys_batched]
  return outs, carry_bdims + ys_bdims

  assert False, "update it"
  consts, init, xs = batched_args
  consts_bdim, init_bdim, xs_bdim = batch_dims

  sizes = _reduce(set.union, map(batching.dimsize, batch_dims, batched_args))
  size = sizes.pop()
  assert not sizes

  consts_batched = batching.where_batched(consts_bdim)
  init_batched = batching.where_batched(init_bdim)
  xs_batched = batching.where_batched(xs_bdim)

  carry_batched = init_batched
  for _ in range(1000):
    which_batched = (consts_batched, carry_batched, xs_batched)
    jaxpr_batched, batched_out = batching.batch_jaxpr(jaxpr, size, which_batched,
                                                      instantiate=(carry_batched, False))
    carry_batched_out, ys_batched = batched_out
    if _binary_lattice_eq(carry_batched_out, carry_batched):
      break
    else:
      carry_batched = _binary_lattice_join(carry_batched_out, carry_batched)
  else:
    raise FixedPointError

  consts_batched = batching.instantiate_bdim(size, 0, consts_batched, consts_bdim, consts)
  init_batched = batching.instantiate_bdim(size, 0, carry_batched, init_bdim, init)
  xs_batched = batching.instantiate_bdim(size, 1, xs_batched, xs_bdim, xs)

  carry_out, ys = scan_p.bind(
      consts_batched, init_batched, xs_batched,
      forward=forward, length=length, jaxpr=jaxpr_batched)

  carry_out_bdim = batching.bools_to_bdims(0, carry_batched)
  ys_bdim = batching.bools_to_bdims(1, ys_batched)
  return core.pack((carry_out, ys)), (carry_out_bdim, ys_bdim)


def scan_bind(*args, **kwargs):
  forward, length = kwargs.pop("forward"), kwargs.pop("length")
  num_consts, num_carry = kwargs.pop("num_consts"), kwargs.pop("num_carry")
  jaxpr, linear = kwargs.pop("jaxpr"), kwargs.pop("linear")
  assert not kwargs
  assert len(linear) == len(args)

  # check that args match input types
  consts, init, xs = split_list(args, [num_consts, num_carry])
  consts_avals, init_avals, x_avals = split_list(jaxpr.in_avals, [num_consts, num_carry])
  xs_avals = map(partial(_promote_aval_rank, length), x_avals)
  assert all(map(core.typecheck, consts_avals, consts))
  assert all(map(core.typecheck, init_avals, init))
  assert all(map(core.typecheck, xs_avals, xs))

  # check that output carry type matches input carry type
  carry_avals, _ = split_list(jaxpr.out_avals, [num_carry])
  # assert init_avals == carry_avals  # TODO carry can be more specific than init

  # check that the data flow is sensible
  core.check_jaxpr(jaxpr.jaxpr)

  return core.Primitive.bind(scan_p, *args, forward=forward, length=length,
                             jaxpr=jaxpr, num_consts=num_consts,
                             num_carry=num_carry, linear=linear)

scan_p = core.Primitive("scan")
scan_p.multiple_results = True
scan_p.def_custom_bind(scan_bind)
scan_p.def_impl(_scan_impl)
ad.primitive_jvps[scan_p] = _scan_jvp
ad.primitive_transposes[scan_p] = _scan_transpose
pe.custom_partial_eval_rules[scan_p] = _scan_partial_eval
xla.initial_style_translations[scan_p] = xla.lower_fun(_scan_impl, initial_style=True)
batching.primitive_batchers[scan_p] = _scan_batching_rule
