from collections import defaultdict
from itertools import count
import numpy as np

primitive_vjps = defaultdict(dict)
def defvjp(fun, *vjps, **kwargs):
    argnums = kwargs.get('argnums', count())
    for argnum, vjp in zip(argnums, vjps):
        primitive_vjps[fun][argnum] = vjp
