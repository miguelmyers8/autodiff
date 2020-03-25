import sys
from . import numpy_wrapper as anp
import numpy as _np
from autograd.tracer import primitive, _container
from autograd.core import backward as _backward
from autograd.core import VJPNode

def ensure_Container(val):
    if isinstance(val, container):
        return val
    else:
        return Container(val)

def set_val(self,val):
    self._value = val
    return self

class container(_container):
    __slots__ = []
    __array_priority__ = 100.0

    @primitive
    def __getitem__(A, idx): return A[idx]

# Constants w.r.t float data just pass though
    shape = property(lambda self: self._value.shape)
    ndim  = property(lambda self: self._value.ndim)
    size  = property(lambda self: self._value.size)
    dtype = property(lambda self: self._value.dtype)
    T = property(lambda self: anp.transpose(self._value))
    def __len__(self): return len(self._value)
    def astype(self, *args, **kwargs): return set_val(self,anp._astype(self._value, *args, **kwargs))

    def __neg__(self): return anp.negative(self)
    def __add__(self, other): return anp.add(     self, other)
    def __sub__(self, other): return anp.subtract(self, other)
    def __mul__(self, other): return anp.multiply(self, other)
    def __pow__(self, other): return anp.power   (self, other)
    def __div__(self, other): return anp.divide(  self, other)
    def __mod__(self, other): return anp.mod(     self, other)
    def __truediv__(self, other): return anp.true_divide(self, other)
    def __matmul__(self, other): return anp.matmul(self, other)
    def __radd__(self, other): return anp.add(     other, self)
    def __rsub__(self, other): return anp.subtract(other, self)
    def __rmul__(self, other): return anp.multiply(other, self)
    def __rpow__(self, other): return anp.power(   other, self)
    def __rdiv__(self, other): return anp.divide(  other, self)
    def __rmod__(self, other): return anp.mod(     other, self)
    def __rtruediv__(self, other): return anp.true_divide(other, self)
    def __rmatmul__(self, other): return anp.matmul(other, self)
    def __eq__(self, other): return anp.equal(self, other)
    def __ne__(self, other): return anp.not_equal(self, other)
    def __gt__(self, other): return anp.greater(self, other)
    def __ge__(self, other): return anp.greater_equal(self, other)
    def __lt__(self, other): return anp.less(self, other)
    def __le__(self, other): return anp.less_equal(self, other)
    def __abs__(self): return anp.abs(self)
    def __hash__(self): return id(self)

    def backward(self):
        _backward(self)


container.register(_np.ndarray)
for type_ in [float, _np.float64, _np.float32, _np.float16,
              complex, _np.complex64, _np.complex128]:
    container.register(type_)


# These numpy.ndarray methods are just refs to an equivalent numpy function
nondiff_methods = ['all', 'any', 'argmax', 'argmin', 'argpartition',
                   'argsort', 'nonzero', 'searchsorted', 'round']
diff_methods = ['clip', 'compress', 'cumprod', 'cumsum', 'diagonal',
                'max', 'mean', 'min', 'prod', 'ptp', 'ravel', 'repeat',
                'reshape', 'squeeze', 'std', 'sum', 'swapaxes', 'take',
                'trace', 'transpose', 'var']

for method_name in nondiff_methods + diff_methods:
    setattr(container, method_name, anp.__dict__[method_name])


# Flatten has no function, only a method.
setattr(container, 'flatten', anp.__dict__['ravel'])

def Container(val,requires_grad=False):
    return container(val,requires_grad=requires_grad,_node=VJPNode.new_root()).astype("float32")
