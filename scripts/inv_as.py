from faultSlip.inversion import Inversion
from faultSlip.catalog_generator import Genrate_catalog
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.spatial import cKDTree
from scipy.optimize import nnls



catalog = np.load('/Users/yohai/workspace/faultSlip/etas/syn_catalog.npy')
gc = Genrate_catalog('/Users/yohai/workspace/faultSlip/etas/syn_catalog.npy',
                     '/Users/yohai/workspace/faultSlip/etas/syn_Gf.npy',
                     '/Users/yohai/workspace/faultSlip/etas/syn_Gf_kdtree.pickel',
                     '/Users/yohai/workspace/faultSlip/etas/syn_slip_Gf.npy',
                     3.0 / 365.0, k=2.84e-3, p=1.07, c=1.78e-5, m_min=2.5)


inv = Inversion('/Users/yohai/workspace/faultSlip/catalog_generator_test.json')
inv.solution = np.load('/Users/yohai/workspace/faultSlip/etas/syn_slip.npy')



# gc.calc_Gf(inv, '/Users/yohai/workspace/faultSlip/etas/syn_Gf')
# gc.calc_slip_Gf(inv, '/Users/yohai/workspace/faultSlip/etas/syn_slip_Gf.npy')
# exit(0)
beta = 0.2
ker = gc.calc_kernal(inv)
ker = np.concatenate((ker, inv.S * beta), axis=0)
as_c = gc.as_in_disloc(inv, 3.0)
as_e = gc.as_contribution(inv)
b = as_c - as_e
b[b < 0] = 0
b = b ** (1.5)
b = np.concatenate((b, np.zeros(600)))
sol = nnls(ker, b)[0]


# inv.solution = np.concatenate((ker[20], ker[345]))
fig, ax = inv.plot_sources(cmap_max=1.0)
ax.scatter(catalog[:,0], catalog[:,1], -catalog[:,2], s=1, color='w')

inv.solution = np.concatenate((sol, np.zeros_like(sol)))
fig, ax = inv.plot_sources(cmap_max=1.0)
ax.scatter(catalog[:,0], catalog[:,1], -catalog[:,2], s=1, color='w')
#
# inv.solution = np.concatenate((as_c - as_e, np.zeros_like(as_e)))
# fig, ax = inv.plot_sources(cmap_max=5.0)
# ax.scatter(catalog[:,0], catalog[:,1], -catalog[:,2], s=1, color='w')

plt.show()