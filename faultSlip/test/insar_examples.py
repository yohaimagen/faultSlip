import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlb
from matplotlib.pylab import *
import sys
sys.path.append('../../')
from faultSlip.inversion import Inversion


def plot_s(
        ax,
        inv,
        movment,
        plot_color_bar=True,
        cmap_max=1.0,
        cmap_min=0.0,
        cmap="jet",
        title="",
        I=-1,
):

    my_cmap = cm.get_cmap(cmap)
    norm = mlb.colors.Normalize(cmap_min, cmap_max)
    shift = 0
    for p in inv.plains:
        p.plot_sources(
            movment[shift: shift + len(p.sources)], ax, my_cmap, norm
        )
        shift += len(p.sources)

    ax.set_title(title)
    if  plot_color_bar:
        cmmapable = cm.ScalarMappable(norm, my_cmap)
        cmmapable.set_array(np.linspace(cmap_min, cmap_max))
        cbar = plt.colorbar(cmmapable)
        cbar.set_label("slip [m]")

inv = Inversion('pars.json')
ss = np.array([0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0])
ds = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
los_dis, uE, uN, uZ = inv.images[0].full_model(ss, ds, 0.25, 0, inv.plains)
print(np.nanmin(los_dis), np.nanmax(los_dis))
# for disp, name in zip((los_dis, uE, uN, uZ), ('los_dis', 'uE', 'uN', 'uZ')):
#     cmap_s = plt.cm.jet
#     norm_s = mlb.colors.Normalize(0, 5)
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1, projection="3d")
#     ax.view_init(15, 300)
#     ax.set_xlabel('E')
#     ax.set_ylabel('N')
#     ax.set_zlabel('Z')
#     inv.plains[0].plot_sources(ss, ax, my_cmap=cmap_s, norm=norm_s)
#     cmap = plt.cm.jet
#     norm = mlb.colors.Normalize(vmin=-0.1, vmax=0.1)
#     colors = cmap(norm(disp))
#     x, y = np.meshgrid(np.linspace(0, 100, inv.images[0].disp.shape[1]), np.linspace(0, 100, inv.images[0].disp.shape[0]))
#     ax.plot_surface(x, y, np.zeros_like(x), cstride=1, rstride=1, facecolors=colors, shade=False)
    # cmmapable = cm.ScalarMappable(norm, cmap)
    # cmmapable.set_array(np.linspace(-0.1, 0.1))
    # cbar = plt.colorbar(cmmapable)
    # cbar.set_label("slip [m]")
    # plt.savefig(f"/Users/yohai/workspace/geodetics_class/figs/s90d60_{name}.png", bbox_inches='tight')
# plt.figure()
# Dunw = ((los_dis * np.pi * 4) / (0.056))
# intf = np.mod(Dunw, 2 * np.pi) - np.pi
# x, y = np.linspace(0, 100, inv.images[0].disp.shape[1]), np.linspace(0, 100, inv.images[0].disp.shape[0])
# plt.pcolormesh(x, y, los_dis, cmap='jet', vmin=-0.1, vmax=0.1)
# # cbar = plt.colorbar()
# # cbar.set_label('LOS displacement')
# plt.savefig(f"/Users/yohai/workspace/geodetics_class/figs/full_res.png", bbox_inches='tight')
# np.save('./los_disp.npy', los_dis)
# plt.show()

inv.images[0].quadtree(0.005, 0.2)
inv.images[0].calculate_station_disp()
print(len(inv.images[0].station))
x = [st.east for st in inv.images[0].station]
y = [st.north for st in inv.images[0].station]
disp = [st.disp for st in inv.images[0].station]
plt.scatter(x, y, c=disp, cmap='jet', vmin=-0.1, vmax=0.1, s=2)
plt.show()
a = 3