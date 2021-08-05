import sys
import matplotlib.pyplot as plt
import matplotlib as mlb
from matplotlib.pylab import *

import numpy as np
from os.path import expanduser
home = expanduser("~")
root = '%s/workspace/thesis/' %home
sys.path.append(root)
from faultSlip.inversion import Inversion
from ridgecrest import solve_CA_cs_no_constraints, solve_CA_cs
from faultSlip.gps_set import Gps
import pickle


def mw(m):
    return (np.log10(m) - 9.05) / 1.5






inv = Inversion('/Users/yohai/workspace/faultSlip/temp.json')
inv.compute_mean = True


inv.calculate_station_disp()
inv.drop_zero_station()
inv.drop_crossing_station()
inv.build_kers()
inv.build_sources_mat()
print(inv.get_sources_num())

# inv.to_gmt('/Users/yohai/workspace/etas/dislocation.gmt', np.zeros(inv.get_sources_num()))
# inv.plot_sources()
# plt.show()
# exit(0)


G_kw = {'beta':0.01, 'alpha':642.0, 'im2sar':10.0}

inv.solve_g(solve_CA_cs, G_kw)


inv.plot_sources()

inv.assign_slip()

n = np.sum([len(p.sources) for p in inv.plains])
# inv.plot_sources_CA_cs(epicenter=(-117.599, 35.77, -8.0))

sol_t = inv.solution
MS = sol_t[0: n]
FS = sol_t[n: n *2]
print(f'fs moment: {mw(inv.seismic_moment(solution=np.concatenate((FS, np.zeros_like(FS)))))}')
print(f'ms moment: {mw(inv.seismic_moment(solution=np.concatenate((MS, np.zeros_like(MS)))))}')

# for i in range(len(inv.images[0].plains)):
#     inv.images[0].to_gmt('/Users/yohai/workspace/CA/plains_plot/test/plain%d.gmt' %i, FS, plains=[i])
# for i in range(len(inv.images[0].plains)):
#     inv.images[0].to_gmt('/Users/yohai/workspace/CA/plains_plot/ms/plain%d.gmt' % i, MS, plains=[i])




inv.gps[0].plot_res(MS)
inv.gps[1].plot_res(FS)
with open('ridgecrest_model.pickle', 'wb') as f:
    pickle.dump(inv, f)
gps = Gps('~/temp.csv')
inv.gps.append(gps)
inv.build_kers()
inv.gps[2].plot_res(MS + FS)
inv.assign_slip()
plt.show()

inv.gps[2].save_model(MS + FS, "/Users/yohai/workspace/hector/caldera_ridgcrest.csv")

# inv.sol_to_geojson(images=inv.images[0:2], path='/Users/yohai/workspace/CA/data_model_res/sar')
b_sar1, G_sar1 = inv.images[0].get_image_kersNdata()
b_sar2, G_sar2 = inv.images[1].get_image_kersNdata()
G_sar = np.concatenate((G_sar1, G_sar2), axis=0)
G_sar = np.concatenate((G_sar, G_sar), axis=1)
vmax = 0.5
inv.plot_sol_val(figsize=6, vmin=-vmax, vmax=vmax, images=inv.images[0:2], G=G_sar)
# inv.plot_sol(figsize=6, vmin=-vmax, vmax=vmax, images=inv.images[0:2])
vmax = 1.5
imgary2sar = 10.0
b_spot1_ew, G_spot1_ew = inv.images[2].get_image_kersNdata()
b_spot1_ns, G_spot1_ns = inv.images[3].get_image_kersNdata()

G_spot1 = np.concatenate((G_spot1_ew, G_spot1_ns), axis=0)
G_spot1 = np.concatenate((G_spot1, np.zeros_like(G_spot1)), axis=1)
# inv.sol_to_geojson(images=inv.images[2:4], path='/Users/yohai/workspace/CA/data_model_res/optic', G=G_spot1)
inv.plot_sol_val(figsize=6, vmin=-vmax, vmax=vmax, images=inv.images[2:4], G=G_spot1)
# inv.plot_sol(figsize=6, vmin=-vmax, vmax=vmax, movment=np.concatenate((MS, np.zeros_like(MS))), images=inv.images[2:4])



plt.show()

