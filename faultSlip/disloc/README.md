disloc
======================

disloc is c translation of the original fortran [code](https://www.bosai.go.jp/e/dc3d.html) by Y. Okada clalculating the static deformation due to displacment along a rectangular dislocation embeded inan elastic half space. Translated originaly by Peter Cervelli pararlise and bridge to python and numpy by Yohai Magen 

----------------------------------
# dependesies
    - numpy
    - cython
compile by runnning 
```console
~$ python  setup.py build_ext --inplace
```
following compiling use the code as foolow:

```python 
from disloc.disloc import disloc_1d
from disloc.disloc import disloc_2d
import numpy as np

poisson_ratio = 0.25
x, y = np.linspace(0, 100, 100), np.linspace(0, 100, 100)
X, Y = np.meshgrid(x, y)
#define dislocation
length, width, depth, dip, strike, x_location, y_location, strike_slip, dip_slip, opening = 20, 10, 1, 90, 0, 50, 50, 1, 0, 0
disloc = np.array([length, width, depth, dip, strike, x_location, y_location, strike_slip, dip_slip, opening], dtype="float64")
# for 1d ndarrays
uE = np.zeros(x.shape, dtype="float64")
uN = np.zeros(x.shape, dtype="float64")
uZ = np.zeros(x.shape, dtype="float64")
disloc_1d(uE, uN, uZ, disloc, x, y, poisson_ratio, x.shape[0], 1)
# for 2d ndarrays
uE = np.zeros(X.shape, dtype="float64")
uN = np.zeros(X.shape, dtype="float64")
uZ = np.zeros(X.shape, dtype="float64")
disloc_2d(uE, uN, uZ, disloc, X, Y, poisson_ratio, x.shape[0], 1)
```