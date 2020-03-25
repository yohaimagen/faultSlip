import numpy as np
import matplotlib.pyplot as plt
from faultSlip.inversion import Inversion


inv = Inversion('/Users/yohai/workspace/faultSlip/faultSlip/example/example.json')

inv.calculate_station_disp()
inv.drop_zero_station()
# inv.drop_crossing_station()
inv.build_kers()
inv.build_sources_mat()

vmax=0.3
# inv.plot_stations(vmax=vmax, vmin=-vmax)


inv.solve(beta=0.000001)
inv.solution = np.concatenate((inv.solution, np.zeros_like(inv.solution)))
inv.plot_sources()

inv.solution = np.load('/Users/yohai/workspace/faultSlip/faultSlip/example/sim_slip.npy')
inv.plot_sources()

plt.show()