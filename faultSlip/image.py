from copy import deepcopy

import matplotlib.patches as patches
from matplotlib.pylab import *
from scipy import ndimage

from faultSlip.disloc import disloc
from faultSlip.station import Station


class Image:
    def __init__(
        self,
        disp_file=None,
        lon=None,
        lat=None,
        x_pixel=None,
        y_pixel=None,
        incidence_angle=None,
        azimuth=None,
        station=None,
        station_size=None,
        plains=None,
        origin_x=None,
        origin_y=None,
        mask_fault=False,
        **kwargs
    ):
        self.disp = np.load(disp_file)
        self.lon = lon
        self.lat = lat
        self.x_pixel = x_pixel
        self.y_pixel = y_pixel
        self.im_x_size = x_pixel * self.disp.shape[1]
        self.im_y_size = y_pixel * self.disp.shape[0]
        self.incidence_angle = np.deg2rad(incidence_angle)
        self.azimuth = np.deg2rad(azimuth)
        self.station = []
        self.G_ss = None
        self.G_ds = None
        self.stations_mat = None
        self.origin_x = origin_x
        if self.origin_x is None:
            self.origin_x = 0
        self.origin_y = origin_y
        if self.origin_y is None:
            self.origin_y = 0
        if station == "uniform":
            y_station_num = int(np.rint(self.im_y_size / station_size))
            x_station_num = int(np.rint(self.im_x_size / station_size))
            e = np.linspace(
                self.origin_x,
                self.origin_x + self.im_x_size,
                self.disp.shape[1],
                endpoint=False,
            )
            n = np.linspace(
                self.origin_y,
                self.origin_y + self.im_y_size,
                self.disp.shape[0],
                endpoint=False,
            )

            x = np.linspace(0, e.shape[0] - 1, x_station_num + 1, dtype=np.int)
            y = np.linspace(0, n.shape[0] - 1, y_station_num + 1, dtype=np.int)

            x_steps = e[x]
            y_steps = n[y]
            for i in range(x_steps.shape[0] - 1):
                for j in range(y_steps.shape[0] - 1):
                    self.station.append(
                        Station(
                            x_steps[i],
                            y_steps[j],
                            x_size=x_steps[i + 1] - x_steps[i],
                            y_size=y_steps[j + 1] - y_steps[j],
                        )
                    )
        else:
            # st = pd.read_csv(station, sep=' ', header=None)
            # st = st.as_matrix()
            st = np.load(station)
            self.station = [
                Station(x, y, x_size, y_size, flag=True)
                for x, y, x_size, y_size in zip(st[:, 0], st[:, 1], st[:, 2], st[:, 3])
            ]

        if mask_fault:
            for p in plains:
                # x0 = p.plain_cord[0]
                # y0 = p.plain_cord[1]
                # ccw_to_x_stk = np.pi / 2 - np.deg2rad(p.strike)
                # x1_t = p.plain_cord[0] + np.cos(ccw_to_x_stk) * p.plain_length
                # y1_t = p.plain_cord[1] + np.sin(ccw_to_x_stk) * p.plain_length
                # x1 = x1_t / self.x_pixel
                # y1 = y1_t / self.y_pixel
                # m = (x1-x0)/(y1-y0)
                # n = y0 - m*x0
                m = np.tan(np.deg2rad(450 - p.strike))
                n = p.plain_cord[1] - m * p.plain_cord[0]
                x_start = p.plain_cord[0]
                x_end = x_start + np.cos(np.deg2rad(450 - p.strike)) * p.plain_length
                if x_start > x_end:
                    x_start, x_end = x_end, x_start
                X = np.linspace(
                    x_start,
                    x_end,
                    int(np.abs(x_start / self.x_pixel - x_end / self.x_pixel)) * 3,
                )
                Y = m * X + n
                X = X / self.x_pixel
                Y = Y / self.y_pixel
                X = X.astype(np.int)
                Y = Y.astype(np.int)
                X_t = [X]
                Y_t = [Y]
                for i in range(1, 100):
                    X_t.append(X + i)
                    X_t.append(X - i)
                    Y_t.append(Y)
                    Y_t.append(Y)
                X = np.concatenate(X_t)
                Y = np.concatenate(Y_t)
                mask = np.logical_and(
                    Y >= 0,
                    np.logical_and(
                        Y < self.disp.shape[0],
                        np.logical_and(X >= 0, X < self.disp.shape[1]),
                    ),
                )
                X = X[mask]
                Y = Y[mask]
                self.disp[Y, X] = np.nan

        self.w_normalizer = 0
        self.model = None
        self.clean = None

    def build_kernal(self, ss, ds, ts, plains, poisson_ratio=0.25):
        def build(plain, ss, ds):
            A = np.zeros((len(plain.sources), len(self.station), 3))
            east = np.array([st.east for st in self.station], dtype="float64")
            north = np.array([st.north for st in self.station], dtype="float64")
            for i, sr in enumerate(plain.sources):
                uE = np.zeros(east.shape, dtype="float64")
                uN = np.zeros(east.shape, dtype="float64")
                uZ = np.zeros(east.shape, dtype="float64")
                model = np.array(
                    [
                        sr.length,
                        sr.width,
                        sr.depth,
                        np.rad2deg(sr.dip),
                        np.rad2deg(sr.strike),
                        0,
                        0,
                        ss,
                        ds,
                        0.0,
                    ],
                    dtype="float64",
                )
                disloc.disloc_1d(
                    uE,
                    uN,
                    uZ,
                    model,
                    east - sr.e,
                    north - sr.n,
                    poisson_ratio,
                    east.shape[0],
                    1,
                )
                A[i, :, 0] = uE
                A[i, :, 1] = uN
                A[i, :, 2] = uZ
            x = -np.cos(self.azimuth) * A[:, :, 0]
            y = np.sin(self.azimuth) * A[:, :, 1]
            z = A[:, :, 2] * np.cos(self.incidence_angle)
            return -((x + y) * np.sin(self.incidence_angle) + z)

        G_ss = []
        G_ds = []
        for p in plains:
            if ss * p.strike_element == 0:
                G_ss.append(np.zeros((len(self.station), 0)))
            else:
                G_ss.append(build(p, ss * p.strike_element, 0.0).T)
            if ds == 0:
                G_ds.append(np.zeros((len(self.station), 0)))
            else:
                G_ds.append(build(p, 0.0, ds * p.dip_element).T)
        self.G_ss = np.concatenate(G_ss, axis=1)
        self.G_ds = np.concatenate(G_ds, axis=1)

    def get_G_ss(self):
        return self.G_ss

    def get_G_ds(self):
        return self.G_ds

    def get_ker(self, compute_mean=False):
        G_ss = self.get_G_ss()
        G_ds = self.get_G_ds()
        if compute_mean:
            weights_vec = np.ones_like(self.stations_mat[:, 5] / self.w_normalizer)
            weights_vec = weights_vec.reshape(-1, 1)
            G_ds = G_ds * weights_vec
            G_ss = G_ss * weights_vec
        return np.concatenate([G_ss, G_ds], axis=1)

    def calculate_station_disp(self, S=None, resamp=False):
        mat = []
        w_normalizer = 1.0
        if S is None:
            stations = self.station
        else:
            stations = S
        for s in stations:
            north = s.north - self.origin_y
            east = s.east - self.origin_x
            try:
                s.disp = self.disp[
                    int(np.rint(north / self.y_pixel)),
                    int(np.rint(east / self.x_pixel)),
                ]
            except:
                print("cords not in data:")
                print(
                    int(np.rint(north / self.y_pixel)),
                    int(np.rint(east / self.x_pixel)),
                )
                exit(1)
            s.weight = 1
            mat.append([s.east, s.north, s.x_size, s.y_size, s.disp, s.weight])
        if S is not None:
            mat = np.array(mat)
            return w_normalizer, mat
        else:
            self.w_normalizer = w_normalizer
            self.stations_mat = np.array(mat)

    def calculate_station_disp_weight(self, S=None, w_norm=0):
        w_normalizer = 1  # w_norm
        # N = np.count_nonzero(~np.isnan(self.disp))
        mat = []

        if S is None:
            stations = self.station
        else:
            stations = S
        for s in stations:
            north = s.north - self.origin_y
            east = s.east - self.origin_x
            data = self.disp[
                int((north - s.y_size / 2.0) / self.y_pixel) : int(
                    (north + s.y_size / 2.0) / self.y_pixel
                ),
                int((east - s.x_size / 2.0) / self.x_pixel) : int(
                    (east + s.x_size / 2.0) / self.x_pixel
                ),
            ]
            try:
                s.weight = float(np.count_nonzero(~np.isnan(data))) / data.size
                if s.weight != 0:
                    s.weight = 1.0
            except:
                print(self.disp.shape)
                print(data.shape)
                print(s.north)
                print(
                    int(s.north / self.y_pixel),
                    int((s.north + s.y_size) / self.y_pixel),
                    int(s.east / self.x_pixel),
                    int((s.east + s.x_size) / self.x_pixel),
                )
                exit(1)
            s.disp = np.nanmean(data)
            # w_normalizer += s.weight
            if np.isnan(s.disp):
                s.disp = 0.0
            mat.append([s.east, s.north, s.x_size, s.y_size, s.disp, s.weight])
        if S is not None:
            mat = np.array(mat)
            # mat[:, 5] *= N
            return w_normalizer, mat
        else:
            self.w_normalizer = w_normalizer
            self.stations_mat = np.array(mat)
            # self.stations_mat[:, 5] *= N

    def get_disp(self, compute_mean=False):
        if compute_mean:
            weights_vec = np.ones_like(self.stations_mat[:, 5] / self.w_normalizer)
            return self.stations_mat[:, 4] * weights_vec
        return self.stations_mat[:, 4]

    def drop_crossing_station(self, plains):
        def line_line_cross(x1, y1, x2, y2, x3, y3, x4, y4):
            uA = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / (
                (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            )
            uB = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / (
                (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            )
            return uA >= 0 and uA <= 1 and uB >= 0 and uB <= 1

        def line_rect_cross(lx1, ly1, lx2, ly2, rx1, ry1, rx2, ry2, rx3, ry3, rx4, ry4):
            return (
                line_line_cross(lx1, ly1, lx2, ly2, rx1, ry1, rx2, ry2)
                or line_line_cross(lx1, ly1, lx2, ly2, rx2, ry2, rx3, ry3)
                or line_line_cross(lx1, ly1, lx2, ly2, rx3, ry3, rx4, ry4)
                or line_line_cross(lx1, ly1, lx2, ly2, rx4, ry4, rx1, ry1)
            )

        x_faults, y_faults = self.get_fault(plains, 2)
        new_stations = []
        delete_rows = []
        for i, st in enumerate(self.station):
            rx1 = st.east - st.x_size / 2
            ry1 = st.north + st.y_size / 2
            rx2 = st.east + st.x_size / 2
            ry2 = st.north + st.y_size / 2
            rx3 = st.east + st.x_size / 2
            ry3 = st.north - st.y_size / 2
            rx4 = st.east - st.x_size / 2
            ry4 = st.north - st.y_size / 2
            st_cross = False
            for x_fault, y_fault in zip(x_faults, y_faults):
                st_cross = st_cross or line_rect_cross(
                    x_fault[0],
                    y_fault[0],
                    x_fault[1],
                    y_fault[1],
                    rx1,
                    ry1,
                    rx2,
                    ry2,
                    rx3,
                    ry3,
                    rx4,
                    ry4,
                )
            if not st_cross:
                new_stations.append(st)
            else:
                delete_rows.append(i)
        self.station = new_stations
        self.stations_mat = np.delete(self.stations_mat, delete_rows, axis=0)

    def drop_nan_station(self):
        new_station = []
        delete_rows = []
        for i, s in enumerate(self.station):
            if not np.isnan(s.disp):
                new_station.append(s)
            else:
                delete_rows.append(i)
        self.station = new_station
        self.stations_mat = np.delete(self.stations_mat, delete_rows, axis=0)

    def drop_zero_station(self):
        new_station = []
        delete_rows = []
        for i, s in enumerate(self.station):
            if s.weight != 0:
                new_station.append(s)
            else:
                delete_rows.append(i)
        self.station = new_station
        self.stations_mat = np.delete(self.stations_mat, delete_rows, axis=0)

    def full_model(self, strike_slip, dip_slip, poisson_ratio, zero_pad, plains):
        E, N = np.meshgrid(
            np.linspace(
                self.origin_x,
                self.origin_x + self.im_x_size,
                self.disp.shape[1],
                endpoint=False,
            ),
            np.linspace(
                self.origin_y,
                self.origin_y + self.im_y_size,
                self.disp.shape[0],
                endpoint=False,
            ),
        )
        uE = np.zeros(E.shape, dtype="float64")
        uN = np.zeros(E.shape, dtype="float64")
        uZ = np.zeros(E.shape, dtype="float64")
        i = 0
        for p in plains:
            for s in p.sources:
                uE_t = np.zeros(E.shape, dtype="float64")
                uN_t = np.zeros(E.shape, dtype="float64")
                uZ_t = np.zeros(E.shape, dtype="float64")
                model = np.array(
                    [
                        s.length,
                        s.width,
                        s.depth,
                        np.rad2deg(s.dip),
                        np.rad2deg(s.strike),
                        0,
                        0,
                        strike_slip[i] * p.strike_element,
                        dip_slip[i] * p.dip_element,
                        0.0,
                    ],
                    dtype="float64",
                )
                disloc.disloc_2d(
                    uE_t,
                    uN_t,
                    uZ_t,
                    model,
                    E - s.e,
                    N - s.n,
                    poisson_ratio,
                    E.shape[0] * E.shape[1],
                    1,
                )
                uE += uE_t
                uN += uN_t
                uZ += uZ_t
                i += 1
            i += zero_pad
        los_dis = -(
            np.sin(self.incidence_angle)
            * (-np.cos(self.azimuth) * uE + np.sin(self.azimuth) * uN)
            + np.cos(self.incidence_angle) * uZ
        )
        self.model = los_dis
        return los_dis, uE, uN, uZ

    def full_model_m(self, strike_slip, dip_slip, poisson_ratio, zero_pad, plains):
        E, N = np.meshgrid(
            np.linspace(0, self.im_x_size, self.disp.shape[1], endpoint=False),
            np.linspace(0, self.im_y_size, self.disp.shape[0], endpoint=False),
        )
        uE = np.zeros(E.shape, dtype="float64")
        uN = np.zeros(E.shape, dtype="float64")
        uZ = np.zeros(E.shape, dtype="float64")
        i = 0
        for p in plains:
            for s in p.sources:
                uE_t = np.zeros(E.shape, dtype="float64")
                uN_t = np.zeros(E.shape, dtype="float64")
                uZ_t = np.zeros(E.shape, dtype="float64")
                model = np.array(
                    [
                        s.length,
                        s.width,
                        s.depth,
                        np.rad2deg(s.dip),
                        np.rad2deg(s.strike),
                        0,
                        0,
                        strike_slip[i],
                        dip_slip[i],
                        0.0,
                    ],
                    dtype="float64",
                )
                disloc.disloc_2d_m(
                    uE_t,
                    uN_t,
                    uZ_t,
                    model,
                    E - s.e,
                    N - s.n,
                    poisson_ratio,
                    E.shape[0] * E.shape[1],
                    1,
                )
                uE += uE_t
                uN += uN_t
                uZ += uZ_t
                i += 1
            i += zero_pad
        los_dis = -(
            np.sin(self.incidence_angle)
            * (-np.cos(self.azimuth) * uE + np.sin(self.azimuth) * uN)
            + np.cos(self.incidence_angle) * uZ
        )
        self.model = los_dis
        return los_dis

    def plot_sol(
        self,
        ax1,
        ax2,
        ax3,
        strike_slip,
        dip_slip,
        plains,
        vmin,
        vmax,
        cmap="jet",
        poisson_ratio=0.25,
        zero_pad=0,
    ):
        def m2dd(m, lat=0):
            "meters to decimal degrees."
            return m / 111319.9 / cos(np.deg2rad(lat))

        def tick_to_strings(ticks):
            str_array = []
            for tic in ticks:
                str_array.append("%.2f" % tic)
            return str_array

        model = self.full_model(strike_slip, dip_slip, poisson_ratio, zero_pad, plains)
        X, Y = self.get_fault(plains)
        tic_y = np.linspace(
            self.origin_x, self.origin_x + self.disp.shape[0], 3, dtype=np.int
        )
        tic_x = np.linspace(
            self.origin_y, self.origin_y + self.disp.shape[1], 4, dtype=np.int
        )[1:-1]
        tic_y_label = tick_to_strings(m2dd(tic_y / self.y_pixel, self.lon) + self.lat)
        tic_x_label = tick_to_strings(m2dd(tic_x / self.x_pixel) + self.lon)
        for ax, img in zip(
            (ax1, ax2, ax3), (self.disp, model[0], self.disp - model[0])
        ):
            im = ax.imshow(
                img,
                cmap=cmap,
                aspect=self.y_pixel / self.x_pixel,
                origin="lower",
                vmax=vmax,
                vmin=vmin,
            )
            ax.set_ylim(0, self.disp.shape[0])
            ax.set_xticks(tic_x)
            ax.set_yticks(tic_y)
            ax.set_xticklabels(tic_x_label)
            ax.set_yticklabels(tic_y_label)

        for x, y in zip(X, Y):
            x -= self.origin_x
            y -= self.origin_y
            x /= self.x_pixel
            y /= self.y_pixel
            ax1.plot(x, y, color="g", linewidth=2)
            ax2.plot(x, y, color="g", linewidth=2)
            ax3.plot(x, y, color="g", linewidth=2)
        return im, model

    def plot_sol_val(self, plains, ax1, ax2, ax3, m_disp, vmin, vmax, cmap="jet"):
        def m2dd(m, lat=0):
            "meters to decimal degrees."
            return m / 111319.9 / cos(np.deg2rad(lat))

        def tick_to_strings(ticks):
            str_array = []
            for tic in ticks:
                str_array.append("%.2f" % tic)
            return str_array

        tic_y = np.linspace(
            self.origin_x, self.origin_x + self.disp.shape[0], 3, dtype=np.int
        )
        tic_x = np.linspace(
            self.origin_y, self.origin_y + self.disp.shape[1], 4, dtype=np.int
        )[1:-1]
        tic_y_label = tick_to_strings(m2dd(tic_y / self.y_pixel, self.lon) + self.lat)
        tic_x_label = tick_to_strings(m2dd(tic_x / self.x_pixel) + self.lon)

        disp = np.array([s.disp for s in self.station])
        for ax, d in zip((ax1, ax2, ax3), (disp, m_disp, disp - m_disp)):
            self.plot_stations_val(ax, plains, cmap, vmax, vmin, d)
            # ax.set_ylim(0, self.disp.shape[0])
            # ax.set_xticks(tic_x)
            # ax.set_yticks(tic_y)
            # ax.set_xticklabels(tic_x_label)
            # ax.set_yticklabels(tic_y_label)

    def sol_to_geojson(self, m_disp, path):
        def m2dd(m, lat=0):
            "meters to decimal degrees."
            return m / 111319.9 / cos(np.deg2rad(lat))

        geojson = """
         { "type": "FeatureCollection",
    "features": [
      %s
       ]
     }
"""
        st = """{ "type": "Feature",
         "geometry": {
           "type": "Polygon",
           "coordinates": [
             [ [%f, %f], [%f, %f], [%f, %f],
               [%f, %f], [%f, %f] ]
             ]
         },
         "properties": {
           "disp":%f,
           "model":%f,
           "res":%f
           }
         }"""
        features = []
        disp = np.array([s.disp for s in self.station])
        for s, d, m, d_m in zip(self.station, disp, m_disp, disp - m_disp):
            x1 = self.lon + m2dd((s.east - s.x_size / 2.0) * 1e3, self.lat)
            y1 = self.lat + m2dd((s.north - s.y_size / 2.0) * 1e3)
            x2 = x1
            y2 = self.lat + m2dd((s.north + s.y_size / 2.0) * 1e3)
            x3 = self.lon + m2dd((s.east + s.x_size / 2.0) * 1e3, self.lat)
            y3 = y2
            x4 = x3
            y4 = y1
            features.append(st % (x1, y1, x2, y2, x3, y3, x4, y4, x1, y1, d, m, d_m))

        with open(path, "w") as f:
            f.write(geojson % ",\n".join(features))

    def plot_model(
        self,
        ax,
        strike_slip,
        dip_slip,
        vmin,
        vmax,
        plains,
        cmap="jet",
        poisson_ratio=0.25,
        zero_pad=0,
        img=None,
    ):
        def m2dd(m, lat=0):
            "meters to decimal degrees."
            return m / 111319.9 / cos(np.deg2rad(lat))

        def tick_to_strings(ticks):
            str_array = []
            for tic in ticks:
                str_array.append("%.2f" % tic)
            return str_array

        if img is None:
            img = self.full_model(
                strike_slip, dip_slip, poisson_ratio, zero_pad, plains
            )
        X, Y = self.get_fault(plains)
        tic_y = np.linspace(
            self.disp.shape[0] * 0.1, self.disp.shape[0] * 0.9, 3, dtype=np.int
        )
        tic_x = np.linspace(
            self.disp.shape[1] * 0.2, self.disp.shape[1] * 0.8, 2, dtype=np.int
        )
        tic_y_label = tick_to_strings(m2dd(tic_y / self.y_pixel, self.lon) + self.lat)
        tic_x_label = tick_to_strings(m2dd(tic_x / self.x_pixel) + self.lon)
        im = ax.imshow(
            img,
            cmap=cmap,
            aspect=self.y_pixel / self.x_pixel,
            origin="lower",
            vmax=vmax,
            vmin=vmin,
        )
        ax.set_ylim(0, self.disp.shape[0])
        ax.set_xticks(tic_x)
        ax.set_yticks(tic_y)
        ax.set_xticklabels(tic_x_label)
        ax.set_yticklabels(tic_y_label)

        for x, y in zip(X, Y):
            x /= self.x_pixel
            y /= self.y_pixel
            ax.plot(x, y, color="g", linewidth=2)

        return im

    def plot_stations(self, ax, plains, dots=True, cmap="jet", vmax=0.2, vmin=-0.2):
        ax.imshow(
            self.disp,
            cmap=cmap,
            aspect=self.y_pixel / self.x_pixel,
            origin="lower",
            vmax=vmax,
            vmin=vmin,
            extent=(0, self.im_x_size, 0, self.im_y_size),
        )
        if dots:
            s_x = [s.east - self.origin_x - s.x_size / 2.0 for s in self.station]
            s_y = [s.north - self.origin_y - s.y_size / 2.0 for s in self.station]
            ax.scatter(s_x, s_y, s=1, color="k")
        else:
            for s in self.station:
                ax.add_patch(
                    patches.Rectangle(
                        (
                            s.east - self.origin_x - s.x_size / 2.0,
                            s.north - self.origin_y - s.y_size / 2.0,
                        ),
                        s.x_size,
                        s.y_size,
                        fill=False,
                        ec="k",
                    )
                )
        X, Y = self.get_fault(plains)
        for x, y in zip(X, Y):
            ax.plot(x - self.origin_x, y - self.origin_y, color="g", linewidth=2)

    def plot_stations_val(self, ax, plains, cmap="jet", vmax=0.2, vmin=-0.2, disp=None):
        if disp is None:
            disp = np.array([s.disp for s in self.station])
        my_cmap = cm.get_cmap(cmap)
        norm = matplotlib.colors.Normalize(vmin, vmax)
        for s, d, i in zip(self.station, disp, range(len(self.station))):
            s_color = my_cmap(norm(d))
            ax.add_patch(
                patches.Rectangle(
                    (
                        s.east - self.origin_x - s.x_size / 2.0,
                        s.north - self.origin_y - s.y_size / 2.0,
                    ),
                    s.x_size,
                    s.y_size,
                    facecolor=s_color,
                    ec=s_color,
                )
            )
            # ax.annotate('%d' %i, (s.east - self.origin_x, s.north - self.origin_y), fontsize=6)
        X, Y = self.get_fault(plains)
        for x, y in zip(X, Y):
            ax.plot(x - self.origin_x, y - self.origin_y, color="g", linewidth=2)

    def get_fault(self, plains, sampels=2):
        X = []
        Y = []
        for p in plains:
            x, y = p.get_fault(1.0, 1.0, sampels)
            X.append(x)
            Y.append(y)
        return X, Y

    def compute_station_disp1(self, s, ss, ds, plains, sources_mat, poisson_ratio=0.25):
        sources_num = 0
        for p in plains:
            sources_num += len(p.sources)
        uE = np.zeros(sources_num, dtype="float64")
        uN = np.zeros(sources_num, dtype="float64")
        uZ = np.zeros(sources_num, dtype="float64")
        model = np.copy(sources_mat[:, 0:-4])
        # should change to remove for loop
        model[:, -2] = (
            np.concatenate([np.ones(len(p.sources)) * p.dip_element for p in plains])
            * ds
        )
        model[:, -3] = (
            np.concatenate([np.ones(len(p.sources)) * p.strike_element for p in plains])
            * ss
        )
        model = model.flatten()
        east = s.east - np.copy(sources_mat[:, -4])
        north = s.north - np.copy(sources_mat[:, -3])
        disloc.disloc_1ds(uE, uN, uZ, model, east, north, poisson_ratio, 1, sources_num)
        disp = -(
            (-np.cos(self.azimuth) * uE + np.sin(self.azimuth) * uN)
            * np.sin(self.incidence_angle)
            + uZ * np.cos(self.incidence_angle)
        )
        return disp

    def compute_source_disp(self, sr, ss, ds, zero_padd=0, poisson_ratio=0.25):
        station_num = self.stations_mat.shape[0]
        uE = np.zeros(station_num + zero_padd, dtype="float64")
        uN = np.zeros(station_num + zero_padd, dtype="float64")
        uZ = np.zeros(station_num + zero_padd, dtype="float64")
        east = np.copy(self.stations_mat[:, 0], order="C")
        north = np.copy(self.stations_mat[:, 1], order="C")
        model = np.array(
            [
                sr.length,
                sr.width,
                sr.depth,
                np.rad2deg(sr.dip),
                np.rad2deg(sr.strike),
                0,
                0,
                ss,
                ds,
                0.0,
            ],
            dtype="float64",
            order="C",
        )
        disloc.disloc_1d(
            uE, uN, uZ, model, east - sr.e, north - sr.n, poisson_ratio, station_num, 1
        )
        return -(
            (-np.cos(self.azimuth) * uE + np.sin(self.azimuth) * uN)
            * np.sin(self.incidence_angle)
            + uZ * np.cos(self.incidence_angle)
        )

    def compute_station_disp(self, s, ss, ds, plains):
        sources_num = 0
        for p in plains:
            sources_num += len(p.sources)
        uE = np.zeros(sources_num, dtype="float64")
        uN = np.zeros(sources_num, dtype="float64")
        uZ = np.zeros(sources_num, dtype="float64")
        i = 0
        for plain in plains:
            for sr in plain.sources:
                e = np.zeros(1)
                n = np.zeros(1)
                z = np.zeros(1)
                model = np.array(
                    [
                        sr.length,
                        sr.width,
                        sr.depth,
                        np.rad2deg(sr.dip),
                        np.rad2deg(sr.strike),
                        0,
                        0,
                        ss * p.strike_element,
                        ds * p.dip_element,
                        0.0,
                    ],
                    dtype="float64",
                )
                disloc.disloc_1d(
                    e,
                    n,
                    z,
                    model,
                    np.array([s.east - sr.e]),
                    np.array([s.north - sr.n]),
                    0.25,
                    1,
                    1,
                )
                uE[i] = e[0]
                uN[i] = n[0]
                uZ[i] = z[0]
                i += 1
        return -(
            (-np.cos(self.azimuth) * uE + np.sin(self.azimuth) * uN)
            * np.sin(self.incidence_angle)
            + uZ * np.cos(self.incidence_angle)
        )

    def resample_model(
        self,
        source_plain,
        source_ind,
        mat_ind,
        strike_element,
        dip_element,
        weights,
        poisson_ratio,
        plains,
    ):
        sr = plains[source_plain].sources[source_ind]
        SR = sr.make_new_source()
        if strike_element != 0:
            B_strike = np.delete(self.get_G_ss(), mat_ind, axis=1)
            for new_sr, k in zip(SR, [mat_ind, mat_ind + 1, mat_ind + 2, mat_ind + 3]):
                B_strike = np.insert(
                    B_strike,
                    k,
                    self.compute_source_disp(
                        new_sr,
                        plains[source_plain].strike_element * strike_element,
                        0,
                        poisson_ratio=poisson_ratio,
                    ),
                    axis=1,
                )

        else:
            B_strike = self.get_G_ss()
        if dip_element != 0:
            B_dip = np.delete(self.get_G_ds(), mat_ind, axis=1)
            for new_sr, k in zip(SR, [mat_ind, mat_ind + 1, mat_ind + 2, mat_ind + 3]):
                B_dip = np.insert(
                    B_dip,
                    k,
                    self.compute_source_disp(
                        new_sr,
                        0,
                        plains[source_plain].dip_element * dip_element,
                        poisson_ratio=poisson_ratio,
                    ),
                    axis=1,
                )
        else:
            B_dip = self.get_G_ds()
        G = np.concatenate((B_strike, B_dip), axis=1)
        if weights:
            weights_vec = self.stations_mat[:, 5] / self.w_normalizer
            weights_vec = np.ones_like(weights_vec.reshape(-1, 1))
            G *= weights_vec
        return G

    def resample(
        self,
        station_ind,
        strike_element,
        dip_element,
        plains,
        weights,
        poisson_ratio,
        surces_mat,
    ):
        s = self.station[station_ind]
        S = s.make_new_stations()
        if strike_element != 0:
            B_strike = np.delete(self.get_G_ss(), station_ind, 0)
            for s, k in zip(
                S, [station_ind, station_ind + 1, station_ind + 2, station_ind + 3]
            ):
                B_strike = np.insert(
                    B_strike,
                    k,
                    self.compute_station_disp1(
                        s,
                        strike_element,
                        0,
                        plains,
                        surces_mat,
                        poisson_ratio=poisson_ratio,
                    ),
                    axis=0,
                )
        else:
            B_strike = np.zeros((self.get_G_ss().shape[0] + 3, 0))
        if dip_element != 0:
            B_dip = np.delete(self.get_G_ds(), station_ind, 0)
            for s, k in zip(
                S, [station_ind, station_ind + 1, station_ind + 2, station_ind + 3]
            ):
                B_dip = np.insert(
                    B_dip,
                    k,
                    self.compute_station_disp1(
                        s,
                        0,
                        dip_element,
                        plains,
                        surces_mat,
                        poisson_ratio=poisson_ratio,
                    ),
                    axis=0,
                )
        else:
            B_dip = np.zeros((self.get_G_ds().shape[0] + 3, 0))
        G = np.concatenate((B_strike, B_dip), axis=1)
        if weights:
            w_normalize, t_station_mat = self.calculate_station_disp_weight(
                S, self.w_normalizer - s.weight
            )
            t_station_mat = np.insert(
                np.delete(self.stations_mat, station_ind, 0),
                station_ind,
                t_station_mat,
                0,
            )
            weights_vec = t_station_mat[:, 5] / w_normalize
            weights_vec = np.ones_like(weights_vec.reshape(-1, 1))
            G *= weights_vec
        return G

    def insert_row(self, ind, station, strike_element, dip_element, plains):
        if strike_element != 0:
            self.G_ss = np.delete(self.G_ss, ind, axis=0)
            for i, s in enumerate(station):
                self.G_ss = np.insert(
                    self.G_ss.T,
                    ind + i,
                    self.compute_station_disp(s, strike_element, 0, plains),
                    axis=1,
                ).T
        else:
            self.strike_kernal = np.zeros((self.G_ss.shape[0] + 3, 0))
        if dip_element != 0:
            self.G_ds = np.delete(self.G_ds, ind, axis=0)
            for i, s in enumerate(station):
                self.G_ds = np.insert(
                    self.G_ds.T,
                    ind + i,
                    self.compute_station_disp(s, 0, dip_element, plains),
                    axis=1,
                ).T
        else:
            self.G_ds = np.zeros((self.G_ds.shape[0] + 3, 0))

    def add_new_stations(
        self, indices, strike_element, dip_element, plains, weights=False
    ):
        indices = np.sort(np.array(indices))
        new_stations = list(self.station)
        shift = 0
        for i in indices:
            shifted = i + shift
            s = new_stations[shifted]
            S = s.make_new_stations()
            if weights:
                self.w_normalizer, t_stations_mat = self.calculate_station_disp_weight(
                    S, self.w_normalizer - s.weight
                )
            else:
                self.w_normalizer, t_stations_mat = self.calculate_station_disp(S)
            self.stations_mat = np.insert(
                np.delete(self.stations_mat, shifted, axis=0),
                shifted,
                t_stations_mat,
                axis=0,
            )
            new_stations[shifted] = S[0]
            new_stations.insert(shifted + 1, S[1])
            new_stations.insert(shifted + 2, S[2])
            new_stations.insert(shifted + 3, S[3])
            self.insert_row(shifted, S, strike_element, dip_element, plains)
            shift += 3
        self.station = new_stations

    def add_new_source(
        self,
        plain_inds,
        sources_inds,
        mat_inds,
        strike_element,
        dip_element,
        plains,
        weights=False,
        poisson_ratio=0.25,
    ):
        new_sources = []
        shift_mat = 0
        for k, plain in enumerate(plains):
            x = np.where(plain_inds == k)[0]
            t_sources_inds = sources_inds[x]
            t_mat_inds = mat_inds[x]
            y = np.argsort(t_sources_inds)
            t_sources_inds = t_sources_inds[y]
            t_mat_inds = t_mat_inds[y]
            new_sources.append(list(plain.sources))
            shift = 0
            for i in range(t_mat_inds.shape[0]):
                sr = plain.sources[t_sources_inds[i]]
                SR = sr.make_new_source()
                shifted = t_sources_inds[i] + shift
                shift += 3
                new_sources[k][shifted] = SR[0]
                new_sources[k].insert(shifted + 1, SR[1])
                new_sources[k].insert(shifted + 2, SR[2])
                new_sources[k].insert(shifted + 3, SR[3])
                mat_in = []
                for s in SR:
                    mat_in.append(
                        np.array(
                            [
                                s.length,
                                s.width,
                                s.depth,
                                np.rad2deg(s.dip),
                                np.rad2deg(s.strike),
                                0,
                                0,
                                0,
                                0,
                                0,
                                s.e,
                                s.n,
                                s.x,
                                s.y,
                            ],
                            dtype="float64",
                        )
                    )
                mat_in = np.vstack(mat_in)
                shifted_mat = t_mat_inds[i] + shift_mat
                shift_mat += 3
                self.sources_mat = np.insert(
                    np.delete(self.sources_mat, shifted_mat, axis=0),
                    shifted_mat,
                    mat_in,
                    axis=0,
                )
                plain.insert_column(
                    self.station,
                    shifted,
                    SR,
                    strike_element,
                    dip_element,
                    self.azimuth,
                    self.incidence_angle,
                    poisson_ratio,
                    zero_pad=0,
                )
                self.sources_num += 3
        for j, plain in enumerate(plains):
            plain.sources = new_sources[j]

    def get_image_kersNdata(self):
        G_img = self.get_ker(compute_mean=False)
        b = self.get_disp(False)
        w = np.ones_like(self.stations_mat[:, 5])
        # w = w / np.sum(w)
        G_w = G_img * w.reshape(-1, 1)
        b_w = b * w
        return b_w, G_w

    def stations_to_csv(self, path, index):
        stations_mat = np.array(
            [[s.east, s.north, s.x_size, s.y_size] for s in self.station]
        )
        if path.endswith(".csv"):
            path = path[:-4] + "_image{}".format(index)
        path += ".csv"
        np.savetxt(path, stations_mat, delimiter=" ")

    def uncorelated_noise(self, mu=0, sigma=0.00001):
        return np.random.normal(mu, sigma, size=self.disp.shape)

    def corelated_noise(self, sigma=10, max_noise=0.001, num_of_points=10):
        im = np.zeros(self.disp.shape)
        mask = np.zeros_like(im, dtype=np.bool).flatten()
        mask[:num_of_points] = True
        np.random.shuffle(mask)
        mask = mask.reshape(im.shape)
        points_val = np.random.uniform(-1, 1, num_of_points)
        im[mask] = points_val.reshape(im[mask].shape)
        im = ndimage.gaussian_filter(im, sigma=sigma)
        im /= np.max(np.abs(im))
        im *= max_noise
        return im

    def restore_clean_displacment(self):
        assert (
            self.clean is not None
        ), "add_uncorelated_noise() or add_corelated_noise(), shold be run befor restoring clean displacment"
        self.disp = self.clean

    def quadtree(self, thershold, min_size, max_grad):
        def quad(s):
            dx = s.x_size / 2.0
            dy = s.y_size / 2.0
            north = s.north - self.origin_y
            east = s.east - self.origin_x
            y1 = int((north - dy) / self.y_pixel)
            y2 = int((north + dy) / self.y_pixel)
            x1 = int((east - dx) / self.x_pixel)
            x2 = int((east + dx) / self.x_pixel)
            data = self.disp[y1:y2, x1:x2]
            not_nans = float(np.count_nonzero(~np.isnan(data)))
            if not_nans / data.size < 0.6:
                self.station.append(deepcopy(s))
                return
            grid = np.indices(data.shape)
            nan_mask = data - data
            # y_grid = grid[0] + nan_mask
            # x_grid = grid[1] + nan_mask
            lux = float(np.unique(grid[1][~np.isnan(data)]).size)
            luy = float(np.unique(grid[0][~np.isnan(data)]).size)
            if (
                not_nans > 2
                and lux > 1
                and luy > 1
                and lux / luy < 3
                and lux / luy > 1 / 3
            ):
                std = np.nanstd(data)
                if std < thershold or s.x_size < min_size or s.y_size < min_size:
                    self.station.append(deepcopy(s))
                    return
                else:
                    S = s.make_new_stations()
                    for sn in S:
                        quad(sn)
            else:
                self.station.append(deepcopy(s))
                return

        old_station = self.station
        self.station = []
        for s in old_station:
            quad(s)

    def insert_column(
        self, mat_ind, SR, strike_element, dip_element, poisson_ratio=0.25
    ):
        east = np.array([st.east for st in self.station], dtype="float64")
        north = np.array([st.north for st in self.station], dtype="float64")

        def calc_disp(ss, ds):
            uE = np.zeros((east.shape[0], 4))
            uN = np.zeros((east.shape[0], 4))
            uZ = np.zeros((east.shape[0], 4))
            for i, sr in enumerate(SR):
                ue = np.zeros(east.shape, dtype="float64")
                un = np.zeros(east.shape, dtype="float64")
                uz = np.zeros(east.shape, dtype="float64")
                model = np.array(
                    [
                        sr.length,
                        sr.width,
                        sr.depth,
                        np.rad2deg(sr.dip),
                        np.rad2deg(sr.strike),
                        0,
                        0,
                        ss,
                        ds,
                        0.0,
                    ],
                    dtype="float64",
                )
                disloc.disloc_1d(
                    ue,
                    un,
                    uz,
                    model,
                    east - sr.e,
                    north - sr.n,
                    poisson_ratio,
                    east.shape[0],
                    1,
                )
                uE[:, i] = ue
                uN[:, i] = un
                uZ[:, i] = uz
            x = -np.cos(self.azimuth) * uE
            y = np.sin(self.azimuth) * uN
            z = uZ * np.cos(self.incidence_angle)
            return -((x + y) * np.sin(self.incidence_angle) + z)

        if strike_element != 0:
            self.G_ss = np.insert(
                np.delete(self.G_ss, mat_ind, axis=1),
                mat_ind,
                calc_disp(strike_element, 0).T,
                axis=1,
            )
        if dip_element != 0:
            self.G_ds = np.insert(
                np.delete(self.G_ds, mat_ind, axis=1),
                mat_ind,
                calc_disp(0, dip_element).T,
                axis=1,
            )
