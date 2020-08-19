from .util import subvals, wraps
from collections import defaultdict
import warnings
import numpy as _np
import contextlib

# ================= Needs work ============
#def trace(start_node, fun, x):
    #with trace_stack.new_trace() as t:
        #start_container = new_container(x, t, start_node)
        #end_container = fun(start_container)
        #if is_container(end_container) and end_container._trace == start_box._container:
            #return end_container._value, end_container._node
        #else:
            #warnings.warn("Output seems independent of input.")
            #return end_container, None

#class TraceStack(object):
    #def __init__(self):
        #self.top = -1
    #@contextmanager
    #def new_trace(self):
        #self.top += 1
        #yield self.top
        #self.top -= 1
#trace_stack = TraceStack()
# =========================================

class Config:
    enable_backprop = True
    train = True

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)

def test_mode():
    return using_config('train', False)


def primitive(f_raw):
    @wraps(f_raw)
    def f_wrapped(*args, **kwargs):
        parents = []
        prev = []
        req = []
        for argnum,arg in enumerate(args):
            if is_container(arg):
                prev.append((argnum,arg))
                req.append(arg.requires_grad)
                if arg.requires_grad:
                    parents.append((argnum, arg._node))
        if prev:
            requires_grad = True in req
            argvals = subvals(args, [(argnum_,container._value) for argnum_ , container in prev])
            argnums = tuple(requires_grad_count for requires_grad_count, _ in parents)
            ans = f_wrapped(*argvals, **kwargs)
            if Config.enable_backprop:
                node = type(prev[0][1]._node)(ans, f_wrapped, argvals, kwargs, argnums, [c[1] for c in parents ])
                return new_container(ans,requires_grad,node)
            node = type(prev[0][1]._node).new_root()
            return new_container(ans,False,node)
        else:
            return f_raw(*args, **kwargs)
    f_wrapped.fun = f_raw
    f_wrapped._is_autograd_primitive = True
    return f_wrapped

notrace_primitives = defaultdict(set)
def register_notrace(trace_type, primitive_fun):
    notrace_primitives[trace_type].add(primitive_fun)

def notrace_primitive(f_raw):
    @wraps(f_raw)
    def f_wrapped(*args, **kwargs):
        argvals = map(getval, args)
        return f_raw(*argvals, **kwargs)
    f_wrapped._is_primitive = True
    return f_wrapped

def ensure_array_float32(arrayable) :
    if isinstance(arrayable, (_container, _np.ndarray)):
        return arrayable.astype("float32")
    else:
        return arrayable

class Node(object):
    def __init__(self, value, fun, args, kwargs, parent_argnums, parents):
        assert False

    def initialize_root(self):
        assert False

    @classmethod
    def new_root(cls, *args, **kwargs):
        root = cls.__new__(cls)
        root.initialize_root(*args, **kwargs)
        return root

class _container(object):
    type_mappings = {}
    types = set()

    def __init__(self, value, requires_grad, _node):
        self._value = ensure_array_float32(value)
        self.requires_grad = requires_grad
        self._node = _node

    @property
    def grad(self):
        if self._node.is_leaf:
            return self._node.saved_grad.reshape(*self._value.shape)

    @grad.setter
    def grad(self, x):
        if self._node.is_leaf:
            self._node.saved_grad = x

    def zero_grad(self):
        if self._node.is_leaf:
            self._node.saved_grad = None

    def __bool__(self):
        return bool(self._value)

    __nonzero__ = __bool__

    def __repr__(self):
        return "Container({0}, requires_grad={1})".format( str(self._value), str(self.requires_grad))

    @classmethod
    def register(cls, value_type):
        _container.types.add(cls)
        _container.type_mappings[value_type] = cls
        _container.type_mappings[cls] = cls



_container_type_mappings = _container.type_mappings
def new_container(value,requires_grad,_node):
    try:
        return  _container_type_mappings[type(value)](value,requires_grad,_node) # map all type to container type and call it
    except KeyError:
        raise TypeError("Can't differentiate w.r.t. type {}".format(type(value)))

_container_types = _container.types
is_container  = lambda x: type(x) in _container_types  # almost 3X faster than isinstance(x, container)
getval = lambda x: getval(x._value) if is_container(x) else x
