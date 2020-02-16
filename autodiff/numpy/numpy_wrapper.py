import numpy as _np
import types
from autodiff.tracer import primitive

def wrap_namespace(old, new):
    """
    takes all of kernels funtions and make them a primative (f_wrapped) and available in the global namespace
    """
    function_types = {_np.ufunc, types.FunctionType, types.BuiltinFunctionType}
    for name, obj in old.items():
        if type(obj) in function_types:
            new[name] = primitive(obj)

"""
allows any numerical libary to be wrapped and have excutions traced
ex: pytorch,cupy etc
"""
wrap_namespace(_np.__dict__, globals())
