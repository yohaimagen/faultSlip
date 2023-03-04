import pandas as pd
import numpy as np
from faultSlip.gps_set import Gps
import os
# import pygmt

class Cpt():
    def __init__(self, series, cmap, output='_______temp', background='i', **kwargs):
        pygmt.makecpt(cmap=cmap, series=series, output=output, background=background, **kwargs)
        cpt = pd.read_csv(output, sep='\t', header=None)
        self.cpt_o = cpt.iloc[-3:]
        cpt = cpt[:-3]
        cpt[0] = np.asfarray(cpt[0])
        self.cpt = cpt
        if output == '_______temp':
            os.remove('_______temp')
    def __call__(self, value):
        if value is None:
            return self.cpt_o.iloc[2, 1]
        idx = np.sum(self.cpt[0] < value) - 1
        if idx < 0:
            return self.cpt_o.iloc[0, 1]
        elif idx + 2 > self.cpt.shape[0]:
            return self.cpt_o.iloc[1, 1]
        else:
            return self.cpt.iloc[idx, 1]


class Strain:
    def __init__(self, data, gps_stations, tris, origin_lon=None, origin_lat=None):
        if isinstance(data, pd.DataFrame):
            self.data = data
        else:
            self.data = pd.read_csv(data)
        if isinstance(tris, np.ndarray):
            self.tris = tris
        else:
            self.tris = np.load(tris)
        self.gps = Gps(gps_stations, origin_lon, origin_lat)
        self.polys = [self.gps.data.iloc[tri] for tri in self.tris]

        def station_ker_1d(st, center_x, center_y):
            return np.stack(([1, 0, st.x - center_x, 0, st.y - center_y, st.y - center_y],
                             [0, 1, 0, st.y - center_y, st.x - center_x, st.x - center_x]))

        def poly_straint_ker(poly):
            center_x = poly.x.mean()
            center_y = poly.y.mean()
            ker = [station_ker_1d(st, center_x, center_y) for i, st in poly.iterrows()]
            ker = np.concatenate(ker, axis=0)
            return ker

        self.kers = [poly_straint_ker(poly) for poly in self.polys]
        self.tris_centers = pd.DataFrame(dict(lon=[p.lon.mean() for p in self.polys],
                                              lat=[p.lat.mean() for p in self.polys]))
        self.G_ss = None
        self.G_ds = None
        self.G_o = None
        self.sources_mat = None
        self.origin_lon = origin_lon
        self.origin_lat = origin_lat


    def build_ker(
        self, strike_element, dip_element, open_elemnt, plains, poisson_ratio=0.25
    ):
        self.gps.build_ker(strike_element, dip_element, open_elemnt, plains, poisson_ratio)
        if strike_element == 0:
            self.G_ss = np.zeros((self.data.shape[0] * 3, 0))
        else:
            self.G_ss = self.build_ker_element(self.gps.G_ss)
        if dip_element == 0:
            self.G_ds = np.zeros((self.data.shape[0] * 3, 0))
        else:
            self.G_ds = self.build_ker_element(self.gps.G_ds)
        if open_elemnt == 0:
            self.G_o = np.zeros((self.data.shape[0] * 3, 0))
        else:
            self.G_o = self.build_ker_element(self.gps.G_o)

    def build_ker_element(self, G):
        E = G[:self.gps.data.shape[0]]
        N = G[self.gps.data.shape[0]:self.gps.data.shape[0] * 2]

        def get_strain(e, n):
            results = np.zeros((self.tris.shape[0], 6))
            for i in range(len(self.kers)):
                results[i] = \
                np.linalg.lstsq(self.kers[i], np.stack((e[self.tris[i]], n[self.tris[i]])).T.flatten(), rcond=None)[0]
            return results[:, [2, 3, 4]].T.flatten()


        strain_ker = np.zeros((self.tris.shape[0] * 3, E.shape[1]))
        for dislocation in range(E.shape[1]):
            strain_ker[:, dislocation] = get_strain(E[:, dislocation], N[:, dislocation])
        return strain_ker

    def plot_strain_poly(self, strain=None, cpt=None, fig=None, region=None, frame=None, projection=f"M20"):
        if frame is None:
            frame = ['wsne', 'xa0.5f0.1', 'ya0.5f0.1']
        if strain is None:
            strain = self.data.e11
        xx = strain * 1e6
        if cpt is None:
            maxxx = np.abs(xx).max()
            dcpt = (maxxx * 2) / 1000
            cpt = [-maxxx, maxxx, dcpt]
            cpt = Cpt(cmap="jet", series=cpt, output='__s.cpt', background='o')
        if fig is None:
            fig = pygmt.Figure()
        pygmt.config(MAP_FRAME_TYPE='plain')
        if region is None:
            region = [self.gps.data.lon.min() - 0.1, self.gps.data.lon.max() + 0.1, self.gps.data.lat.min() - 0.1,
                      self.gps.data.lat.max() + 0.1]
        fig.coast(
            frame=frame,
            projection=projection,
            region=region,
            land="gray"
        )
        for i, poly in enumerate(self.polys):
            fig.plot(
                x=poly.lon,
                y=poly.lat,
                color=cpt(xx[i]),
                close=True,
                pen='0.05'
            )
        fig.colorbar(cmap="__s.cpt", frame=["x+lStrain"])
        os.remove('__s.cpt')
        return fig
    def calc_strain_axis(self, strain=None):
        if strain is None:
            strain = self.data
        eigs = []
        angels = []
        for e11, e22, e12 in zip(strain.e11, strain.e22, strain.e12):
            E = np.zeros((2, 2))
            E[0, 0] = e11
            E[1, 1] = e22
            E[1, 0] = e12
            E[0, 1] = E[1, 0]
            eig, V = np.linalg.eig(E)
            V = V[:, np.argsort(eig)[::-1]]
            eigs.append(np.sort(eig)[::-1])
            angels.append(np.degrees(np.arctan2(*V[:, 1])))

        eigs = np.stack(eigs)
        angels = np.array(angels)
        return pd.DataFrame(
            dict(lon=self.tris_centers.lon, lat=self.tris_centers.lat, e1=eigs[:, 0] * 1e6, e2=eigs[:, 1] * 1e6,
                 az=angels))
    def plot_axis(self, fig, strain=None, color='black'):
        if strain is None:
            strain = self.data
        strain_ax = self.calc_strain_axis(strain)
        fig.velo(
            data=strain_ax,
            spec='x0.2c',
            vector=f"4p+p1p,{color}"
        )

    def plot_strain_axis(self, strain=None, fig=None, region=None, frame=None, projection=f"M10"):
        if strain is None:
            strain = self.data
        if frame is None:
            frame = ['wsne', 'xa0.5f0.1', 'ya0.5f0.1']
        if fig is None:
            fig = pygmt.Figure()
        pygmt.config(MAP_FRAME_TYPE='plain')
        if region is None:
            region = [self.gps.data.lon.min() - 0.1, self.gps.data.lon.max() + 0.1, self.gps.data.lat.min() - 0.1,
                      self.gps.data.lat.max() + 0.1]

        fig.coast(
            frame=frame,
            projection=projection,
            region=region,
            land="gray"
        )
        self.plot_axis(fig, strain)
        return fig



