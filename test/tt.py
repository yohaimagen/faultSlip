import numpy as np
import pygmt as pygmt
import xarray as xr
import matplotlib.pyplot as plt
from faultSlip.utils import *
import sys
sys.path.append('/Users/yohai/workspace/okada_source/')
from okada.okada import *

strike, dip, direction = 0,90, 'rl'

X, Y = np.meshgrid(np.arange(0, 101, dtype=np.float), np.arange(0, 101, dtype=np.float))
Z = np.ones_like(X) * 0.1

stress = np.zeros((X.shape[0], X.shape[1], 3, 3))
disp = np.zeros((X.shape[0], X.shape[1], 3))

x = 50.01
y = 50.01
z = 0.2
d_strike = np.radians(strike)
d_dip = np.radians(dip)
L = 10.0
W = 10.0
slip_strike = -1.0
slip_dip = 0.0
s_open = 0.0

r_strike = np.radians(strike)
r_dip = np.radians(dip)
r_rake = np.radians(180)

yang_mod = 8*1e5
poisson = 0.25
lame_lambda = (yang_mod * poisson) / ((1 + poisson) * ( 1 - 2 * poisson))
shear_mod = yang_mod / (2 * (1 + poisson))
lame_lambda, shear_mod

for ix in range(X.shape[0]):
    for jx in range(X.shape[1]):
        u, s = psokada_t(x, y, z, d_strike, d_dip, L, W, slip_strike, slip_dip, s_open, X[ix, jx], Y[ix, jx], Z[ix, jx], lame_lambda, shear_mod)
        stress[ix, jx]  = s
        disp[ix, jx] = u