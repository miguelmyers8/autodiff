import numpy as np

def wraps(fun, namestr="{fun}", docstr="{doc}", **kwargs):
    def _wraps(f):
        try:
            f.__name__ = namestr.format(fun=get_name(fun), **kwargs)
            f.__doc__ = docstr.format(fun=get_name(fun), doc=get_doc(fun), **kwargs)
        finally:
            return f
    return _wraps

def subvals(x, ivs):
    x_ = list(x)
    for i, v in ivs:
        x_[i] = v
    return tuple(x_)

def subval(x, i, v):
    x_ = list(x)
    x_[i] = v
    return tuple(x_)

def toposort(end_conatiner):
    child_counts = {}
    stack = [end_conatiner]
    while stack:
        conatiner = stack.pop()
        if conatiner in child_counts:
            child_counts[conatiner] += 1
        else:
            child_counts[conatiner] = 1
            stack.extend(conatiner.parents)

    childless_conatiners = [end_conatiner]
    while childless_conatiners:
        conatiner = childless_conatiners.pop()
        yield conatiner
        for parent in conatiner.parents:
            if child_counts[parent] == 1:
                childless_conatiners.append(parent)
            else:
                child_counts[parent] -= 1
