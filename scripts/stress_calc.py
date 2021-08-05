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
from osgeo import gdal, osr



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


#inv.plot_sources()

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


inv.solution = np.concatenate((MS + FS, np.zeros_like(MS)))

X1, Y1 = np.meshgrid(np.linspace(0, 100, 800), np.linspace(0, 100, 800))
Z1 = np.ones_like(X1) * 1.0
# plain = inv.images[0].plains[5]
dip = 90#plain.dip
strike = 80#plain.strike
# print plain.dip, plain.strike
a = inv.calc_coulomb(0.6, X1.flatten(), Y1.flatten(), Z1.flatten(), strike=np.deg2rad(strike), dip=np.deg2rad(dip))
print('strike %.2f, dip %.2f' %(strike, dip))
# col = a[5].reshape(X1.shape) / 1000000



vmax = 1e6
fig, axs = plt.subplots(1,1)
axs.imshow(a[4].reshape(X1.shape), origin='lower', cmap='seismic', vmin=-vmax, vmax=vmax)
axs.set_title(r'$\Delta \sigma_{n}$')
axs.set_xlim(60, 150)
axs.set_ylim(60, 140)
axs.set_yticks([])
axs.set_xticks([])

fig, axs = plt.subplots(1, 3)
axs[0].pcolor(X1, Y1, a[3].reshape(X1.shape), cmap='seismic', vmin=-vmax, vmax=vmax)
axs[0].set_title('coulomb')
axs[1].pcolor(X1, Y1, a[4].reshape(X1.shape), cmap='seismic', vmin=-vmax, vmax=vmax)
axs[1].set_title(r'$\Delta \sigma_{n}$')
axs[2].pcolor(X1, Y1, a[5].reshape(X1.shape), cmap='seismic', vmin=-vmax, vmax=vmax)
axs[2].set_title('ts')
for i in range(3):
    X, Y = inv.plains[0].get_fault(1.0, 1.0)
    axs[i].plot(X, Y, color='k')
    X, Y = inv.plains[3].get_fault(1.0, 1.0)
    axs[i].plot(X, Y, color='k')
    X, Y = inv.plains[4].get_fault(1.0, 1.0)
    axs[i].plot(X, Y, color='k')
    X, Y = inv.plains[5].get_fault(1.0, 1.0)
    axs[i].plot(X, Y, color='k')
plt.show()
#
#
lower_l_y = inv.images[0].lat #- Inversion.m2dd(100e3)
lower_l_x = inv.images[0].lon #- Inversion.m2dd(100e3, lat=34.5)
np.save('/Users/yohai/workspace/dep_presentaion/garlock/coulomb.npy', a[3].reshape(X1.shape))
print(lower_l_y, lower_l_x)

def arrray2wgstiff(path, array, lower_left_lon, lower_left_lat, x_dd, y_dd):
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(path, array.shape[1], array.shape[0], 1, gdal.GDT_Float32)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    print(lower_left_lon, x_dd, 0.0, lower_left_lat + y_dd *array.shape[0], 0.0, -y_dd)
    ds.SetGeoTransform((lower_left_lon, x_dd, 0.0, lower_left_lat + y_dd *array.shape[0], 0.0, -y_dd))
    ds.GetRasterBand(1).WriteArray(np.flipud(array))
    ds = None

# arrray2wgstiff('~/workspace/dep_presentaion/garlock/coulomb.tif', a[3].reshape(X1.shape), lower_l_x, lower_l_y, Inversion.m2dd(1e3, 34), Inversion.m2dd(1e3))