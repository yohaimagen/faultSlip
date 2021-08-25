from faultSlip.inversion import Inversion
import matplotlib.pyplot as plt
import numpy as np


inv = Inversion('./__params.json')
inv.build_kers()

def get_G(inv, G_kw):
    return inv.strain[0].G_ds

G_kw = {}
print(inv.compute_cn_g(get_G, G_kw))


inv.plot_sources(view=(30, 290))


cns = np.array(inv.resample_model(get_G, G_kw, 4, 3.0, 1))
inv.build_sources_mat()
inv.save_sources_mat('s_mat')
# G, inv2 = inv.resample_model_s(0, 2, get_G, G_kw)
# np.save('cns.npy', cns)
inv.plot_sources(view=(30, 290))
plt.figure()
plt.plot(cns)
plt.show()

# fig = inv.strain[0].plot_strain_poly()
# fig.show()
