import sys
import os
root = '/Users/yohai/workspace/faultSlip'
sys.path.append(root)
from faultSlip.inversion import Inversion
from faultSlip.utils import *
import matplotlib.pyplot as plt
import numpy as np

os.chdir('/Users/yohai/workspace/faultSlip/test_sample_up')




inv = Inversion('./params.json')
# inv.plot_sources()
# plt.show()
inv.build_kers()
inv.calculate_station_disp()
# inv.drop_zero_station()

def get_G(inv, G_kw):
    G = np.concatenate((inv.images[0].G_ss, inv.images[0].G_ds), axis=1)
    return G

inv.sample_up_model(get_G, {}, 3, 5)

inv.plot_sources()
plt.show()