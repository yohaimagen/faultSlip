import numpy as np
import matplotlib.pyplot as plt
from inversion import Inversion


inv = Inversion('/Users/yohai/workspace/faultSlip/faultSlip/example/resampling_data/resample_data_example.json')

inv.calculate_station_disp()
inv.drop_zero_station()
# inv.drop_crossing_station()
inv.build_kers()
inv.build_sources_mat()

vmax=0.3
inv.plot_stations(vmax=vmax, vmin=-vmax)

def get_G(inv, G_kw):
    return G_kw['kernal_arry'][0]
G_kw = {}

data_points, cn_vec, res = inv.resample_data(get_G, G_kw, N=30, data_per_r=10)

inv.plot_stations(vmax=vmax, vmin=-vmax)
plt.figure()
plt.plot(data_points, cn_vec)
plt.yscale('log')
plt.show()

inv.solve(beta=0.0001)
inv.solution = np.concatenate((inv.solution, np.zeros_like(inv.solution)))
inv.plot_sources()



plt.show()