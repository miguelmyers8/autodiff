# Autodiff (THIS PROJECT IS EXTREMLY BUGGY)
I want to learn autograd the right way<br>
This is a reimplementation/reengineering based on [the full version of Autograd](https://github.com/hips/autograd).

## The goal:
To dive into Matthew Johnson autograd package, understand it the best I can, document, and reimplement.<br>
This autograd will function like pytorch.

## Rescores:
[The video slides by Matthew](http://videolectures.net/deeplearning2017_johnson_automatic_differentiation/)<br>
[the full version of Autograd](https://github.com/hips/autograd)<br>
[tutorial implementation ](https://github.com/mattjj/autodidact)<br>

## Example:
```python
from autograd.numpy.container import container, VJPNode
import numpy as _np
import autograd.numpy as anp

def Container(val,requires_grad=False):
    return container(val,requires_grad=requires_grad,_node=VJPNode.new_root())

x = Container(_np.linspace(-7,7,2),False)
i = Container(_np.linspace(-3,3,2),True)

p = 9+i*x+2/x
anp.sum(p).backward()
i.grad
```
