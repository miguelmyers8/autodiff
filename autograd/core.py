from collections import defaultdict
from autograd.util import toposort
from itertools import count
import numpy as np


def backward(conatiner):
    current_node = conatiner._node
    assert conatiner.requires_grad, "called backward on non-requires-grad tensor"
    #if current_conatiner.shape != ():
        #raise RuntimeError("grad must be specified for non-0-tensor")
    g = np.ones_like(conatiner._value)
    outgrads = {current_node: g}
    for node in toposort(current_node):
        outgrad = outgrads.pop(node)
        if node.is_leaf:
            node.saved_grad = outgrad
        fun, value, args, kwargs, argnums = node.recipe
        for argnum, parent in zip(argnums, node.parents):
            vjp = primitive_vjps[fun][argnum]
            parent_grad = vjp(outgrad, value, *args, **kwargs)

            outgrads[parent] = add_outgrads(outgrads.get(parent), parent_grad)


def add_outgrads(prev_g, g):
    if prev_g is None:
        return g
    return prev_g + g


primitive_vjps = defaultdict(dict)
def defvjp(fun, *vjps, **kwargs):
    argnums = kwargs.get('argnums', count())
    for argnum, vjp in zip(argnums, vjps):
        primitive_vjps[fun][argnum] = vjp
