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


def reduce_to_fault(inv, df):
    mask = np.zeros(df.shape[0], dtype=np.bool)
    X0 = df[['x', 'y', 'depth']].values
    for p in inv.plains:
        for s in p.sources:
            cun_hull = ConvexHull(get_box(s, 3.0))
            for i in range(X0.shape[0]):
                mask[i] = np.logical_or(mask[i], [pnt_in_cvex_hull_1(cun_hull, X0[i])])
    return df[mask]

def get_stress(inv, df_backround, df_event, dim, time_delta_1, time_delta_2):
    num_of_event_backround = []
    num_of_event_event = []
    depth = []
    for p in inv.plains:
        new_sources = []
        for s in p.sources:
            box = get_box(s, dim)
            x, y, z = box[:, 0], box[:, 1], box[:, 2]
            df_backround_t = df_backround[np.logical_and.reduce((df_backround.depth >= z.min(),
                                                                 df_backround.depth <= z.max(), df_backround.x >= x.min(),
                                                                 df_backround.x <= x.max(),df_backround.y >= y.min(),
                                                                 df_backround.y <= y.max()))]
            df_event_t = df_event[np.logical_and.reduce((df_event.depth >= z.min(), df_event.depth <= z.max(),
                                                         df_event.x >= x.min(), df_event.x <= x.max(),
                                                         df_event.y >= y.min(), df_event.y <= y.max()))]
            if df_backround_t.shape[0] > 7 and df_event_t.shape[0] > 7:
                num_of_event_backround.append(df_backround_t.shape[0])
                num_of_event_event.append(df_event_t.shape[0])
                depth.append(np.mean(z))
                new_sources.append(s)
        p.sources = new_sources
    num_of_event_backround = np.array(num_of_event_backround)
    num_of_event_event = np.array(num_of_event_event)
    depth = np.array(depth)
    return 1e-3 * 24e6 * depth * np.log((num_of_event_event / time_delta_2) / (num_of_event_backround / time_delta_1))

def to_csv(plains, stress, path):
    df = []
    i = 0
    for p in plains:
        if p.strike_element == -1:
            rake = np.deg2rad(0)
        else:
            rake = np.deg2rad(180)
        for s in p.sources:
            df.append([s.e_m, s.n_m, s.depth_m, s.strike, s.dip, rake, stress[i]])
            i += 1
    df = pd.DataFrame(df)
    df.columns = ['x', 'y', 'z', 'strike', 'dip', 'rake', 'ds']
    df.to_csv(path, index=False)

