import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/yohai/workspace/faultSlip')
from faultSlip.inversion import Inversion
import os 
os.chdir('/Users/yohai/workspace/faultSlip/faultSlip/example/simple_inversion/')

inv = Inversion('/Users/yohai/workspace/faultSlip/faultSlip/example/simple_inversion/example.json')

inv.calculate_station_disp()
# inv.drop_zero_station()
# inv.drop_crossing_station()
inv.build_kers()
inv.build_sources_mat()

vmax=0.3
# inv.plot_stations(vmax=vmax, vmin=-vmax)


# inv.solve(beta=0.000001)
inv.solution = np.concatenate((np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 2, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]), np.zeros(25)))
inv.plot_sources(view=(90, 30))

# inv.solution = np.load('/Users/yohai/workspace/faultSlip/faultSlip/example/sim_slip.npy')
# inv.plot_sources()

plt.show()