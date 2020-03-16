from __future__ import absolute_import
import numpy as onp
from . import numpy_wrapper as anp
from .conatiner import Conatiner
from autograd.tracer import primitive
from autograd.extend import defvjp



# ----- Binary ufuncs -----

defvjp(anp.add,         lambda ans, x, y : unbroadcast_f(x, lambda g: g),
                        lambda ans, x, y : unbroadcast_f(y, lambda g: g))
defvjp(anp.multiply,    lambda ans, x, y : unbroadcast_f(x, lambda g: y * g),
                        lambda ans, x, y : unbroadcast_f(y, lambda g: x * g))
defvjp(anp.subtract,    lambda ans, x, y : unbroadcast_f(x, lambda g: g),
                        lambda ans, x, y : unbroadcast_f(y, lambda g: -g))
defvjp(anp.divide,      lambda ans, x, y : unbroadcast_f(x, lambda g:   g / y),
                        lambda ans, x, y : unbroadcast_f(y, lambda g: - g * x / y**2))
defvjp(anp.maximum,     lambda ans, x, y : unbroadcast_f(x, lambda g: g * balanced_eq(x, ans, y)),
                        lambda ans, x, y : unbroadcast_f(y, lambda g: g * balanced_eq(y, ans, x)))
defvjp(anp.minimum,     lambda ans, x, y : unbroadcast_f(x, lambda g: g * balanced_eq(x, ans, y)),
                        lambda ans, x, y : unbroadcast_f(y, lambda g: g * balanced_eq(y, ans, x)))
defvjp(anp.fmax,        lambda ans, x, y : unbroadcast_f(x, lambda g: g * balanced_eq(x, ans, y)),
                        lambda ans, x, y : unbroadcast_f(y, lambda g: g * balanced_eq(y, ans, x)))
defvjp(anp.fmin,        lambda ans, x, y : unbroadcast_f(x, lambda g: g * balanced_eq(x, ans, y)),
                        lambda ans, x, y : unbroadcast_f(y, lambda g: g * balanced_eq(y, ans, x)))
defvjp(anp.logaddexp,   lambda ans, x, y : unbroadcast_f(x, lambda g: g * anp.exp(x-ans)),
                        lambda ans, x, y : unbroadcast_f(y, lambda g: g * anp.exp(y-ans)))
defvjp(anp.logaddexp2,  lambda ans, x, y : unbroadcast_f(x, lambda g: g * 2**(x-ans)),
                        lambda ans, x, y : unbroadcast_f(y, lambda g: g * 2**(y-ans)))
defvjp(anp.true_divide, lambda ans, x, y : unbroadcast_f(x, lambda g: g / y),
                        lambda ans, x, y : unbroadcast_f(y, lambda g: - g * x / y**2))
defvjp(anp.mod,         lambda ans, x, y : unbroadcast_f(x, lambda g: g),
                        lambda ans, x, y : unbroadcast_f(y, lambda g: -g * anp.floor(x/y)))
defvjp(anp.remainder,   lambda ans, x, y : unbroadcast_f(x, lambda g: g),
                        lambda ans, x, y : unbroadcast_f(y, lambda g: -g * anp.floor(x/y)))
defvjp(anp.power,
    lambda ans, x, y : unbroadcast_f(x, lambda g: g * y * x ** anp.where(y, y - 1, 1.)),
    lambda ans, x, y : unbroadcast_f(y, lambda g: g * anp.log(replace_zero(x, 1.)) * ans))

def replace_zero(x, val):
    return anp.where(x, x, val)

def unbroadcast(x, target_meta, broadcast_idx=0):
    target_shape, target_ndim, dtype, target_iscomplex = target_meta
    while anp.ndim(x) > target_ndim:
        x = anp.sum(x, axis=broadcast_idx)
    for axis, size in enumerate(target_shape):
        if size == 1:
            x = anp.sum(x, axis=axis, keepdims=True)
    if anp.iscomplexobj(x) and not target_iscomplex:
        x = anp.real(x)
    return x


def unbroadcast_f(target, f):
    target_meta = anp.metadata(target)
    return lambda g: unbroadcast(f(g), target_meta)

def balanced_eq(x, z, y):
    return (x == z) / (1.0 + (x == y))
# ----- Simple grads -----

defvjp(anp.negative, lambda ans, x: lambda g: -g)
defvjp(anp.exp,    lambda ans, x : lambda g: ans * g)



defvjp(anp.where, None,
       lambda ans, c, x=None, y=None : lambda g: anp.where(c, g, anp.zeros(g.shape)),
       lambda ans, c, x=None, y=None : lambda g: anp.where(c, anp.zeros(g.shape), g))

defvjp(anp.reshape, lambda ans, x, shape, order=None : lambda g: anp.reshape(g, anp.shape(x), order=order))
