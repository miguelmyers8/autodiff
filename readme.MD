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
from autograd.numpy.container import Container
import numpy as _np
import autograd.numpy as anp

x = Container(_np.linspace(-7,7,2),False)
i = Container(_np.linspace(-3,3,2),True)

p = 9+i*x+2/x
p.backward()
i.grad
```
