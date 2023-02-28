import sys
import os
root = '/Users/yohai/workspace/faultSlip'
sys.path.append(root)
from faultSlip.inversion import Inversion
from faultSlip.utils import *
import matplotlib.pyplot as plt
import numpy as np

os.chdir('/Users/yohai/workspace/faultSlip/test_ptofile_2/')




# fig, ax = plt.subplots(1, 1)
# inv.plot_profiles_2_location(ax=ax)
inv = Inversion('./pars.json')

inv.build_kers()

sol = np.ones(sum([len(p.sources) for p in inv.plains])) * 3
sol = np.concatenate((sol, np.zeros_like(sol)))
for prof in inv.profiles_2:
    prof.plot_model(slip=sol)


# inv.quad_profiles_2(0.03, 1)
# inv.build_kers()

# sol = np.ones(sum([len(p.sources) for p in inv.plains])) * 3
# sol = np.concatenate((sol, np.zeros_like(sol)))
# for prof in inv.profiles_2:
#     prof.plot_model(slip=sol)

inv.plot_profiles_2_location()

plt.show()



# inv.plot_sources_2d()
# plt.show()