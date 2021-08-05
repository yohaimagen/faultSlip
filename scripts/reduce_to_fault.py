import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from faultSlip.inversion import Inversion
from scipy.spatial import ConvexHull

from source import Source


def get_line_pt(x, y, z, length, ang):
    dx = np.cos(ang) *length
    dy = np.sin(ang) * length
    return x - dx, y - dy, z, x + dx, y + dy, z
def get_box(s, dim):
    ulx, uly, uld, urx, ury, urd = get_line_pt(s.e_t, s.n_t, s.depth_t, s.length/2 + dim, s.ccw_to_x_stk)
    dlx, dly, dld, drx, dry, drd = get_line_pt(s.e, s.n, s.depth, s.length / 2 + dim, s.ccw_to_x_stk)
    x1, y1, z1, x2, y2, z2 = get_line_pt(ulx, uly, uld, dim, s.ccw_to_x_dip)
    x3, y3, z3, x4, y4, z4 = get_line_pt(urx, ury, urd, dim, s.ccw_to_x_dip)
    x5, y5, z5, x6, y6, z6 = get_line_pt(dlx, dly, dld, dim, s.ccw_to_x_dip)
    x7, y7, z7, x8, y8, z8 = get_line_pt(drx, dry, drd, dim, s.ccw_to_x_dip)
    x = np.array([x1, x2, x3, x4, x5, x6, x7, x8]).reshape(-1, 1)
    y = np.array([y1, y2, y3, y4, y5, y6, y7, y8]).reshape(-1, 1)
    z = np.array([z1, z2, z3, z4, z5, z6, z7, z8]).reshape(-1, 1)
    return np.concatenate((x, y, z), axis=1)
def pnt_in_cvex_hull_1(hull, pnt):
    '''
    Checks if `pnt` is inside the convex hull.
    `hull` -- a QHull ConvexHull object
    `pnt` -- point array of shape (3,)
    '''
    new_hull = ConvexHull(np.concatenate((hull.points, [pnt])))
    if np.array_equal(new_hull.vertices, hull.vertices):
        return True
    return False
def in_disloc_range(s, dim, pnt):
    cun_hull = ConvexHull(get_box(s, dim))
    return pnt_in_cvex_hull_1(cun_hull, pnt)



inv = Inversion('/Users/yohai/workspace/faultSlip/temp2.json')
df = pd.read_csv('/Users/yohai/workspace/CA/seismisty/fs_seismisity.csv', sep=' ', header=None)
print(df.shape)
mask = np.logical_and(df[8] > -117.75, np.logical_and(df[8] < -117.33, np.logical_and(df[7] > 35.4, df[7] < 36.0)))
df = df[mask]
print(df.shape)
df['x'] = Inversion.dd2m(df[8] + 118.1, df[7])*1e-3
df['y'] = Inversion.dd2m(df[7] - 35.3)*1e-3
fault_x, fault_y = inv.get_fault()
plt.scatter(df.x, df.y, s=1)
for x, y in zip(fault_x, fault_y):
    plt.plot(x, y, color='k', linewidth=2)

mask = np.zeros(df.shape[0], dtype=np.bool)
X0 = df[['x', 'y', 9]].values
for p in inv.plains:
    for s in p.sources:
        cun_hull = ConvexHull(get_box(s, 3.0))
        for i in range(X0.shape[0]):
            mask[i] = np.logical_or(mask[i], [pnt_in_cvex_hull_1(cun_hull, X0[i])])

df = df[mask]
plt.scatter(df.x, df.y, s=1)
for x, y in zip(fault_x, fault_y):
    plt.plot(x, y, color='k', linewidth=2)
print(df.shape)
df.to_csv('/Users/yohai/workspace/CA/seismisty/fs_seismisity_temp.csv', sep=' ', header=False, index=False)
plt.show()


df = pd.read_csv('/Users/yohai/workspace/CA/seismisty/background_seismisity_tt.csv', sep=' ', header=None)
df['x'] = Inversion.dd2m(df[8] + 118.1, df[7])*1e-3
df['y'] = Inversion.dd2m(df[7] - 35.3)*1e-3
print(df.shape)


fs_df = pd.read_csv('/Users/yohai/workspace/CA/seismisty/background_seismisity_temp.csv', sep=' ', header=None)
fs_df['x'] = Inversion.dd2m(fs_df[8] + 118.1, fs_df[7])*1e-3
fs_df['y'] = Inversion.dd2m(fs_df[7] - 35.3)*1e-3
print(fs_df.shape)

ms_df = pd.read_csv('/Users/yohai/workspace/CA/seismisty/ms_seismisity_tt.csv', sep=' ', header=None)
ms_df['x'] = Inversion.dd2m(ms_df[8] + 118.1, ms_df[7])*1e-3
ms_df['y'] = Inversion.dd2m(ms_df[7] - 35.3)*1e-3
print(ms_df.shape)


fig, ax = inv.plot_sources()
ax.scatter(df.x, df.y, -df[9], s=1, color='b')
ax.scatter(fs_df.x, fs_df.y, -fs_df[9], s=1, color='y')
ax.scatter(ms_df.x, ms_df.y, -ms_df[9], s=1, color='g')
fig, ax = inv.plot_sources()
ax.scatter(df.x, df.y, -df[9], s=1, color='b')
fig, ax = inv.plot_sources()
ax.scatter(fs_df.x, fs_df.y, -fs_df[9], s=1, color='y')
fig, ax = inv.plot_sources()
ax.scatter(ms_df.x, ms_df.y, -ms_df[9], s=1, color='g')



plt.show()