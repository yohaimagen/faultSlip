import numpy as np
import matplotlib.pyplot as plt
from inversion import Inversion


inv = Inversion('/Users/yohai/workspace/faultSlip/faultSlip/example/Ridgecrest/Ridgecrest.json')

inv.calculate_station_disp()
inv.drop_zero_station()
inv.build_kers()
inv.build_sources_mat()

vmax=0.3
inv.plot_stations(vmax=vmax, vmin=-vmax)
inv.plot_sources()
def get_G(inv, G_kw):
    return np.concatenate(G_kw['kernal_arry'], axis=0)
G_kw = {}

iteration, num_of_sources, num_of_stations, cn = inv.combine_resample(get_G, G_kw, 70, 2.0, 1.0, 10)

inv.plot_stations(vmax=vmax, vmin=-vmax)
plt.figure()
inv.plot_sources()
plt.figure()
plt.plot(iteration, cn)
# plt.yscale('log')
plt.show()

inv.S = inv.new_smoothing()
inv.solve(beta=0.05)
inv.solution = np.concatenate((inv.solution, np.zeros_like(inv.solution)))
inv.plot_sources()

# inv.plot_sol_val()



plt.show()