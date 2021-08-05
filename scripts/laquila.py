import numpy as np
import pandas as pd
from inversion import Full_inversion

def build_offset_constrain(self, offset_data, plain_ind):
    offset_df = pd.read_csv(offset_data)

    x0, y0 = self.images[0].plains[plain_ind].plain_cord[0], self.images[0].plains[plain_ind].plain_cord[1]
    offset_df['x'] = Full_inversion.dd2m(offset_df.xcoord - self.images[0].lon, lat=self.images[0].lat)
    offset_df['y'] = Full_inversion.dd2m(
        offset_df.ycoord - (self.images[0].lat - Full_inversion.m2dd(self.images[0].disp.shape[0] * self.images[0].y_pixel * 1e3)))
    m = -np.tan(np.deg2rad(self.images[0].plains[plain_ind].strike - np.pi / 2))
    n = y0 - m * x0
    offset_df['x_p'] = (offset_df.y * 1e-3 * m + offset_df.x * 1e-3 - m * n) / (m ** 2 + 1)
    offset_df['y_p'] = m * offset_df.x_p + n
    offset_df['d'] = np.sqrt((offset_df.x_p - x0) ** 2 + (offset_df.y_p - y0) ** 2)

    mask = []
    e = []
    n = []
    length = []
    plain = []
    depth = []
    s_in_plains = []
    for i, p in enumerate(self.images[0].plains):
        s_in_plains.append(0)
        for s in p.sources:
            s_in_plains[i] += 1
            plain.append(i)
            mask.append(s.depth_t == 0)
            e.append(s.e_t)
            n.append(s.n_t)
            length.append(s.length)
            depth.append(s.depth_t)
    sources_df = pd.DataFrame(np.vstack((plain, mask, depth, e, n, length)).T,
                              columns=['plain', 'surface', 'depth', 'e', 'n', 'length'])
    sources_df['d'] = np.sqrt((sources_df.e - x0) ** 2 + (sources_df.n - y0) ** 2)
    sources_df['ds'] = sources_df.d - sources_df.length / 2
    sources_df['de'] = sources_df.d + sources_df.length / 2

    offset_s = []
    for i, row in sources_df.iterrows():
        if row.plain == plain_ind and row.surface == 1:
            offset_s.append(np.mean(offset_df[(offset_df.d > row.ds) & (offset_df.d < row.de)]['offset']))
        else:
            offset_s.append(np.nan)
    sources_df['offset'] = offset_s
    b = []
    # c = np.zeros((np.sum(np.logical_not(np.isnan(sources_df.offset.values))), sources_df.shape[0]))
    c = np.zeros((int(np.sum(sources_df.surface == 1)), sources_df.shape[0]))
    j = 0
    for i, row in sources_df.iterrows():
        if np.isnan(row.offset):
            if row.surface == 1:
                b.append(0)
                c[j, i] = 1
                j += 1
        else:
            b.append(row.offset)
            c[j, i] = 1
            j += 1
    c = np.concatenate((np.zeros_like(c), c, np.zeros((int(np.sum(sources_df.surface)), 2))), axis=1)
    return c, np.array(b)