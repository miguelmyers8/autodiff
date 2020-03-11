from .util import subvals, wraps

def primitive(f_raw):
    def f_wrapped(*args, **kwargs):
        parents = []
        prev = []
        req = []
        for argnum,arg in enumerate(args):
            if is_conatiner(arg):
                prev.append((argnum,arg))
                req.append(arg.requires_grad)
                if arg.requires_grad:
                    parents.append((argnum, arg._node))
        if prev:
            requires_grad = True in req
            argvals = subvals(args, [(argnum_,conatiner._value) for argnum_ , conatiner in prev])
            argnums = tuple(requires_grad_count for requires_grad_count, _ in parents)
            ans = f_raw(*argvals, **kwargs)
            node = type(prev[0][1]._node)(ans, f_wrapped, argvals, kwargs, argnums, [c[1] for c in parents ])
            return new_conatiner(ans,node,requires_grad)
        else:
            return f_raw(*args, **kwargs)
    return f_wrapped


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

class _conatiner(object):
    type_mappings = {}
    types = set()

    def __init__(self, value, _node,requires_grad=False):
        self._value = value
        self._node = _node
        self.requires_grad = requires_grad

    @property
    def grad(self):
        if self._node.is_leaf:
            return self._node.saved_grad

    def __bool__(self):
        return bool(self._value)

    __nonzero__ = __bool__

    def __repr__(self):
        return "Conatiner({0}, requires_grad={1})".format( str(self._value), str(self.requires_grad))

    @classmethod
    def register(cls, value_type):
        _conatiner.types.add(cls)
        _conatiner.type_mappings[value_type] = cls
        _conatiner.type_mappings[cls] = cls



_conatiner_type_mappings = _conatiner.type_mappings
def new_conatiner(value,requires_grad,_node):
    try:
        return  _conatiner_type_mappings[type(value)](value,requires_grad,_node) # map all type to conatiner type and call it
    except KeyError:
        raise TypeError("Can't differentiate w.r.t. type {}".format(type(value)))

_conatiner_types = _conatiner.types
is_conatiner  = lambda x: type(x) in _conatiner_types  # almost 3X faster than isinstance(x, conatiner)
getval = lambda x: getval(x._value) if isconatiner(x) else x
