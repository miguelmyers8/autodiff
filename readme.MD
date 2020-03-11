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
from autograd.numpy.conatiner import Conatiner
import numpy as _np
import autograd.numpy as anp

x = Conatiner(_np.linspace(-7,7,2),False)
i = Conatiner(_np.linspace(-3,3,2),True)

p = i*x+2
p.backward()
i.grad
```
