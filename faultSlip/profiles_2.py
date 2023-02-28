import numpy as np
from faultSlip.disloc import disloc
from faultSlip.utils import dd2m
import matplotlib.pyplot as plt
import matplotlib as mlb
import pyproj

class Profile_2(object):
    """
    A class representing a displacment profile.

    Attributes:
    x (np.ndarray): x coordinates of the displacment profile
    y (np.ndarray): y coordinates of the displacment profile
    data (np.ndarray): displacment data
    heading (float): profile heading in degrees
    incidence_angle (float): profile incidence angle in degrees
    n_hat (np.ndarray): unit vector of the displacment profile
    """

    def __init__(self, origin_lon, origin_lat, lon, lat, data, heading, uncertentices=None):
        self.full_lon = np.load(lon)
        self.full_lat = np.load(lat)
        geodesic = pyproj.Geod(ellps='WGS84') 
        _, _, x = geodesic.inv(self.full_lon, self.full_lat, np.ones_like(self.full_lon) * origin_lon, self.full_lat)
        self.full_x = x * 1e-3
        self.full_x[self.full_lon < origin_lon] *= -1
        # self.full_x = dd2m(self.full_lon - origin_lon, np.mean(self.full_lat)) * 1e-3
        _, _, y = geodesic.inv(self.full_lon, self.full_lat, self.full_lon, np.ones_like(self.full_lon) * origin_lat)
        self.full_y = y * 1e-3
        self.full_y[self.full_lat < origin_lat] *= -1
        # self.full_y = dd2m(self.full_lat - origin_lat) * 1e-3
        self.full_data = np.load(data)
        self.lon = np.copy(self.full_lon)
        self.lat = np.copy(self.full_lat)
        self.data = np.copy(self.full_data)
        self.x = np.copy(self.full_x)
        self.y = np.copy(self.full_y)
        self.heading = np.radians(heading)
        self.n_hat = np.array([np.sin(self.heading), np.cos(self.heading), 0])
        if uncertentices is not None:
            self.uncertentices = np.load(uncertentices)
        else:
            self.uncertentices = None

    def build_ker(
        
        self, strike_element, dip_element, open_elemnt, plains, poisson_ratio=0.25
    ):
        '''
        The build_ker function builds the kernel matrix of a profile object given a set of strike, dip, and open  elements and a poisson ratio. It returns the kernel matrix.

        Args:

    strike_element: float, the strike element(1, -1, 0)
        dip_element: float, the dip element(1, -1, 0)
        open_elemnt: float, the open element(1, -1, 0)
        plains: list of Plain objects, the plains in the model
        poisson_ratio: float, the poisson ratio of the material
        Returns:

        G_ss: 2d numpy array, the strike slip kernel matrix
        G_ds: 2d numpy array, the dip slip kernel matrix
        G_o: 2d numpy array, the open kernel matrix
        '''
        if strike_element == 0:
            self.G_ss = np.zeros((self.data.shape[0], 0))
        else:
            self.G_ss = self.build_ker_element(
                strike_element, 0, 0, plains, poisson_ratio
            )
        if dip_element == 0:
            self.G_ds = np.zeros((self.data.shape[0], 0))
        else:
            self.G_ds = self.build_ker_element(0, dip_element, 0, plains, poisson_ratio)
        if open_elemnt == 0:
            self.G_o = np.zeros((self.data.shape[0], 0))
        else:
            self.G_o = self.build_ker_element(0, 0, open_elemnt, plains, poisson_ratio)

    def build_ker_element(
        self, strike_element, dip_element, open_element, plains, poisson_ratio=0.25
    ):
        """
        Build a kernel matrix for the profile data

        Args:
        strike_element: Strike element(1, -1, 0)
        dip_element: Dip element(1, -1, 0)
        open_element: Open element(1, -1, 0)
        plains: list of Plain objects, the plains in the model
        poisson_ratio: Poisson's ratio (default: 0.25)

        Returns:
        The kernel matrix for the profile data
        """
        all_Gz = []
        all_Ge = []
        all_Gn = []
        for plain in plains:
            s_element = strike_element * plain.strike_element
            d_element = dip_element * plain.dip_element
            o_element = open_element * plain.open_element
            if np.all(np.array([s_element, d_element, o_element]) == 0):
                Gz = np.zeros((self.data.shape[0], 0))
                Ge = np.zeros_like(Gz)
                Gn = np.zeros_like(Gz)
            else:
                Gz = np.zeros((self.data.shape[0], len(plain.sources)))
                Ge = np.zeros_like(Gz)
                Gn = np.zeros_like(Gz)
                for i, sr in enumerate(plain.sources):
                    uE = np.zeros(self.data.shape[0], dtype="float64")
                    uN = np.zeros_like(uE)
                    uZ = np.zeros_like(uE)
                    model = np.array(
                        [
                            sr.length,
                            sr.width,
                            sr.depth,
                            np.rad2deg(sr.dip),
                            np.rad2deg(sr.strike),
                            0,
                            0,
                            s_element,
                            d_element,
                            o_element,
                        ],
                        dtype="float64",
                    )
                    disloc.disloc_1d(
                        uE,
                        uN,
                        uZ,
                        model,
                        self.x - sr.e,
                        self.y - sr.n,
                        poisson_ratio,
                        self.data.shape[0],
                        1,
                    )
                    Gz[:, i] = uZ - uZ[-1]
                    Ge[:, i] = uE - uE[-1]
                    Gn[:, i] = uN - uN[-1]
            all_Ge.append(Ge)
            all_Gn.append(Gn)
            all_Gz.append(Gz)
        G = np.stack(
            (
                np.concatenate(all_Ge, axis=1),
                np.concatenate(all_Gn, axis=1),
                np.concatenate(all_Gz, axis=1),
            )
        )
        G = G.T.dot(self.n_hat.reshape(-1, 1)).squeeze().T
        if len(G.shape) == 1:
            G = G.reshape(-1, 1)
        return G

    def quadtree(self, thershold, min_size):
        def add_data(data, lon, lat, x, y):
            new_data.append(np.nanmedian(data))
            new_x.append(np.nanmean(x))
            new_y.append(np.nanmean(y))
            new_lon.append(np.nanmean(lon))
            new_lat.append(np.nanmean(lat))


        def quad(dist, data, lon, lat, x, y):
            length = dist[-1] - dist[0]
            new_length = length / 2
            mask_l = dist > new_length
            mask_r = dist < new_length
            not_nans = np.sum(~np.isnan(data[mask_l]))
            std = np.nanstd(data[mask_l])
            if not_nans / data.size < 0.1 or std < thershold or np.abs(new_length) < min_size:
                add_data(data[mask_l], lon[mask_l], lat[mask_l], x[mask_l], y[mask_l])
            else:
                quad(dist[mask_l] - dist[mask_l][0], data[mask_l], lon[mask_l], lat[mask_l], x[mask_l], y[mask_l])

            not_nans = np.sum(~np.isnan(data[mask_r]))
            std = np.nanstd(data[mask_r])
            if not_nans / data.size < 0.1 or std < thershold or np.abs(new_length) < min_size:
                add_data(data[mask_r], lon[mask_r], lat[mask_r], x[mask_r], y[mask_r])
            else:
                quad(dist[mask_r] - dist[mask_r][0], data[mask_r], lon[mask_r], lat[mask_r], x[mask_r], y[mask_r])

        new_x = []
        new_y = []
        new_lon = []
        new_lat = []
        new_data = []
        dist = (self.x**2 + self.y**2)**0.5
        dist -= dist[0]
        quad(dist, self.data, self.lon, self.lat, self.x, self.y)
        idxs = np.argsort(new_x)
        self.data = np.array(new_data)[idxs]
        self.x = np.array(new_x)[idxs]
        self.y = np.array(new_y)[idxs]
        self.lon = np.array(new_lon)[idxs]
        self.lat = np.array(new_lat)[idxs]

    def midfilt(self, size=5):
        cut = -(self.full_data.shape[0] % size)
        if cut == 0:
            cut = self.full_data.shape[0]
        self.data = np.nanmedian(self.full_data[:cut].reshape(-1, size), axis=1)
        self.x = np.nanmean(self.full_x[:cut].reshape(-1, size), axis=1)
        self.y = np.nanmean(self.full_y[:cut].reshape(-1, size), axis=1)
        self.lon = np.nanmean(self.full_lon[:cut].reshape(-1, size), axis=1)
        self.lat = np.nanmean(self.full_lat[:cut].reshape(-1, size), axis=1)

    def plot_profile(self, ax=None):
        """
        Plots a profile of the data.

        Args:
        ax (matplotlib.Axes): Axis to plot on. If None, creates a new one.

        Returns:
        matplotlib.Axes: Axis on which the profile is plotted.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        dist = (self.x**2 + self.y**2)**0.5
        dist -= dist[0]
        ax.scatter(dist, self.data, s=3, color='k')
        if self.uncertentices is not None:
            ax.errorbar(dist, self.data, yerr=self.uncertentices, color='r', linestyle="None")
        return ax

    def get_model(self, slip):
        G = np.concatenate((self.G_ss, self.G_ds, self.G_o), axis=1)
        m = G.dot(slip.reshape(-1, 1))
        return m

    def plot_model(self, slip, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        m = self.get_model(slip)
        ax.scatter(self.full_x, self.full_data, s=1, color='k')
        ax.scatter(self.x, self.data)
        ax.scatter(self.x, m, color='r')

    def plot_location(self, ax=None, vmin=-1, vmax=1, cmap='jet', d=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        if d is None:
            d = self.data
        ax.scatter(self.x, self.y, c=d, cmap=cmap, s=2, vmin=vmin, vmax=vmax)
        return ax