from faultSlip.inversion import Inversion
import matplotlib.pyplot as plt
import numpy as np


inv = Inversion('./__t.json')
inv.build_kers()
inv.solution = np.array([1.0, 0.0])


X, Y = np.meshgrid(np.linspace(0, 100, 10), np.linspace(0, 100, 10))
Z = np.ones_like(X) * 0.1

stress = inv.calc_coulomb_2d(0.4, X, Y, Z, 0, 90, 0, 30e9, 30e9, )