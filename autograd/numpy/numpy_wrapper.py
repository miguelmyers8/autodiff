import numpy as _np
import types
from autograd.tracer import primitive

def wrap_namespace(old, new):

    function_types = {_np.ufunc, types.FunctionType, types.BuiltinFunctionType}
    for name, obj in old.items():
        if type(obj) in function_types:
            new[name] = primitive(obj)


@primitive
def _astype(A, dtype, order='K', casting='unsafe', subok=True, copy=True):
  return A.astype(dtype, order, casting, subok, copy)


wrap_namespace(_np.__dict__, globals())
