import numpy as np
from faultSlip.disloc import disloc
import matplotlib.pyplot as plt
class Profile(object):
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

    def __init__(self, x, y, data, heading, incidence_angle, uncertentices=None):
        self.x = np.load(x)
        self.y = np.load(y)
        self.data = np.load(data)
        self.heading = np.radians(heading)
        self.incidence_angle = np.radians(incidence_angle)
        self.n_hat = np.array([np.cos(self.incidence_angle) * np.sin(self.heading), np.cos(self.incidence_angle) * np.cos(self.heading), np.sin(self.incidence_angle)])
        if uncertentices is not None:
            self.uncertentices = np.load(uncertentices)

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
        dist = (self.x**2 + self.y**2)**0.5
        dist -= dist[0]
        ax.plot(dist, m, color='r')
    def plot_location(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.plot(self.x, self.y)
        return ax