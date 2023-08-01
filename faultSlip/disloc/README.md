# disloc
======================

disloc is a C translation of the original Fortran  [code](https://www.bosai.go.jp/e/dc3d.html) by Y. Okada. It calculates the static deformation due to displacement along a rectangular dislocation embedded in an elastic half-space. The code was initially translated by Peter Cervelli, then parallelized and bridged to Python and NumPy by Yohai Magen.

----------------------------------
# dependesies
    - numpy
    - cython
### Compile the code by running the following command in your terminal:

```console
~$ python  setup.py build_ext --inplace
```
# Usage
### After compiling the code, you can use it as follows:

```python 
from disloc.disloc import disloc_1d
from disloc.disloc import disloc_2d
import numpy as np

# Define parameters
poisson_ratio = 0.25
x, y = np.linspace(0, 100, 100), np.linspace(0, 100, 100)
X, Y = np.meshgrid(x, y)

# Define dislocation
length, width, depth, dip, strike, x_location, y_location, strike_slip, dip_slip, opening = 20, 10, 1, 90, 0, 50, 50, 1, 0, 0
disloc = np.array([length, width, depth, dip, strike, x_location, y_location, strike_slip, dip_slip, opening], dtype="float64")

# For 1D ndarrays
uE = np.zeros(x.shape, dtype="float64")
uN = np.zeros(x.shape, dtype="float64")
uZ = np.zeros(x.shape, dtype="float64")
disloc_1d(uE, uN, uZ, disloc, x, y, poisson_ratio, x.shape[0], 1)

# For 2D ndarrays
uE = np.zeros(X.shape, dtype="float64")
uN = np.zeros(X.shape, dtype="float64")
uZ = np.zeros(X.shape, dtype="float64")
disloc_2d(uE, uN, uZ, disloc, X, Y, poisson_ratio, x.shape[0], 1)
```
### The above code creates a grid of points (x, y), defines a dislocation, and then uses the disloc_1d and disloc_2d functions to calculate the displacement at each point due to the dislocation. The displacement is calculated in the east (uE), north (uN), and vertical (uZ) directions.