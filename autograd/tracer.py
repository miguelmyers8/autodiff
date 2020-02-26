from .util import subvals


def primitive(f_raw):
    def f_wrapped(*args, **kwargs):
        boxed_args, trace_id = find_top_boxed_args(args)
        argvals = subvals(args, [(argnum, box._value) for argnum, box in boxed_args])
        ans = new_box(f_raw(*argvals, **kwargs))
        return ans
    return f_wrapped

def find_top_boxed_args(args):
    top_trace_id = -1
    top_boxes = []
    for argnum, arg in enumerate(args):
        if is_node(arg):
            top_boxes.append((argnum, arg))
    return top_boxes, top_trace_id


class _node(object):
    type_mappings = {}
    types = set()

    def __init__(self, value, requires_grad=False):
        self._value = value
        self.requires_grad = requires_grad

    def __bool__(self):
        return bool(self._value)

    __nonzero__ = __bool__

    def __repr__(self):
        return "Node({0}, requires_grad={1})".format( str(self._value), str(self.requires_grad))
        
    @classmethod
    def register(cls, value_type):
        _node.types.add(cls)
        _node.type_mappings[value_type] = cls
        _node.type_mappings[cls] = cls



_node_type_mappings = _node.type_mappings
def new_box(value):
    try:
        return _node_type_mappings[type(value)](value) # map all type to box type and call it
    except KeyError:
        raise TypeError("Can't differentiate w.r.t. type {}".format(type(value)))

_node_types = _node.types
is_node  = lambda x: type(x) in _node_types  # almost 3X faster than isinstance(x, Box)
getval = lambda x: getval(x._value) if isbox(x) else x
