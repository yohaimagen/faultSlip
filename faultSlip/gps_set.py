import pandas as pd
from matplotlib.pylab import *

from faultSlip.disloc import disloc


class Gps:
    def __init__(self, data, origin_lon=None, origin_lat=None):
        """
        Initialize the `Gps` object.

        Args:
            data (pd.DataFrame): The GPS data.
            origin_lon (float): The longitude of the origin of the coordinate system.
            origin_lat (float): The latitude of the origin of the coordinate system.

        """
        if isinstance(data, pd.DataFrame):
            self.data = data
        else:
            self.data = pd.read_csv(data)
        self.G_ss = None
        self.G_ds = None
        self.G_o = None
        self.sources_mat = None
        self.origin_lon = origin_lon
        self.origin_lat = origin_lat

    def build_ker(
        
        self, strike_element, dip_element, open_elemnt, plains, poisson_ratio=0.25
    ):
    
        if strike_element == 0:
            self.G_ss = np.zeros((self.data.shape[0] * 3, 0))
        else:
            self.G_ss = self.build_ker_element(
                strike_element, 0, 0, plains, poisson_ratio
            )
        if dip_element == 0:
            self.G_ds = np.zeros((self.data.shape[0] * 3, 0))
        else:
            self.G_ds = self.build_ker_element(0, dip_element, 0, plains, poisson_ratio)
        if open_elemnt == 0:
            self.G_o = np.zeros((self.data.shape[0] * 3, 0))
        else:
            self.G_o = self.build_ker_element(0, 0, open_elemnt, plains, poisson_ratio)

    def build_ker_element(
        self, strike_element, dip_element, open_element, plains, poisson_ratio=0.25
    ):
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
                        self.data.x.values * 1e-3 - sr.e,
                        self.data.y.values * 1e-3 - sr.n,
                        poisson_ratio,
                        self.data.shape[0],
                        1,
                    )
                    Gz[:, i] = uZ
                    Ge[:, i] = uE
                    Gn[:, i] = uN
            all_Ge.append(Ge)
            all_Gn.append(Gn)
            all_Gz.append(Gz)
        xx = np.concatenate(
            (
                np.concatenate(all_Ge, axis=1),
                np.concatenate(all_Gn, axis=1),
                np.concatenate(all_Gz, axis=1),
            ),
            axis=0,
        )
        return xx

    def cala_whigts(self, mask=None):
        sigma = np.concatenate(
            (self.data.Se.values, self.data.Sn.values, self.data.Su.values)
        )
        if mask is not None:
            sigma = sigma[mask]
        return 1 / (sigma * np.sum(1 / sigma[sigma != 0]))

    def softmax_whigts(self, mask=None):
        sigma = np.concatenate(
            (self.data.Se.values, self.data.Sn.values, self.data.Su.values)
        )
        sigma = np.exp(sigma)
        return sigma / np.sum(sigma)

    def get_data(self):
        return np.concatenate(
            (self.data.E.values, self.data.N.values, self.data.Up.values)
        )

    def plot_res(self, slip):
        G = np.concatenate((self.G_ss, self.G_ds), axis=1)
        model_d = G.dot(slip.reshape(-1, 1))
        shift = 0
        plt.figure()
        plt.quiver(self.data.x, self.data.y, self.data.E, self.data.N)
        em = model_d[: self.data.shape[0]]
        nm = model_d[self.data.shape[0]: self.data.shape[0] * 2]
        plt.quiver(self.data.x, self.data.y, em, nm, color='r')

        plt.figure()
        plt.quiver(self.data.x, self.data.y, np.zeros_like(self.data.E), self.data.Up)
        um = nm = model_d[self.data.shape[0] * 2:]
        plt.quiver(self.data.x, self.data.y, np.zeros_like(em), um, color='r')

        # for d, err in zip(["E", "N", "Up"], ["Se", "Sn", "Su"]):
        #     plt.figure()
        #     plt.bar(np.arange(self.data.shape[0]), self.data[d], yerr=self.data[err])
        #     y = model_d[shift : shift + self.data.shape[0]]
        #     plt.bar(np.arange(self.data.shape[0]), y.flatten(), width=0.4)
        #     shift += self.data.shape[0]
        #     plt.ylabel("displacment [mm]")
        #     plt.xlabel("GPS station")
        #     plt.legend()

    def calc_misfit(self, slip):
        G = np.concatenate((self.G_ss, self.G_ds), axis=1)
        model_d = G.dot(slip)
        obs = np.concatenate((self.data["E"], self.data["N"], self.data["Up"]))
        return np.linalg.norm(obs - model_d) / np.linalg.norm(obs)

    def calc_rms(self, slip):
        G = np.concatenate((self.G_ss, self.G_ds), axis=1)
        model_d = G.dot(slip[:-2])
        obs = np.concatenate((self.data["E"], self.data["N"], self.data["Up"]))
        return np.sqrt((1.0 / obs.shape[0]) * np.sum(np.power(obs - model_d, 2))) * 1000

    def plot_config(self):
        plt.figure()
        plt.scatter(self.data.x * 1e-3, self.data.y * 1e-3, color="k", s=2)
        if "id" in self.data.columns:
            for i, txt in enumerate(self.data.id):
                plt.annotate(txt, (self.data.x[i] * 1e-3, self.data.y[i] * 1e-3))
        X, Y = self.get_fault()
        for x, y in zip(X, Y):
            plt.plot(x, y, color="g")

    def save_model(self, slip, path=None, only_gps=True):
        G = np.concatenate((self.G_ss, self.G_ds, self.G_o), axis=1)
        if only_gps:
            model_d = G.dot(slip.reshape(-1, 1))
        else:
            model_d = G.dot(slip[:-2].reshape(-1, 1))
        model = self.data[["id", "lon", "lat"]].copy()
        model.loc[:, "E"] = model_d[0 : self.data.shape[0]]
        model.loc[:, "N"] = model_d[self.data.shape[0] : self.data.shape[0] * 2]

        model.loc[:, "Up"] = model_d[self.data.shape[0] * 2 : self.data.shape[0] * 4]
        if path is not None:
            model.to_csv(path, index=False)
        return model

    def get_model(self, slip):
        G = np.concatenate((self.G_ss, self.G_ds), axis=1)
        model_d = G.dot(slip.reshape(-1, 1))
        model = self.data[["id", "lon", "lat"]].copy()
        model.loc[:, "E"] = model_d[0 : self.data.shape[0]]
        model.loc[:, "N"] = model_d[self.data.shape[0] : self.data.shape[0] * 2]
        model.loc[:, "Up"] = model_d[self.data.shape[0] * 2 : self.data.shape[0] * 4]
        return model

    def get_model_EN(self, slip):
        G = np.concatenate((self.G_ss, self.G_ds), axis=1)
        model_d = G.dot(slip.reshape(-1, 1))
        return np.concatenate(
            (
                model_d[0 : self.data.shape[0]],
                model_d[self.data.shape[0] : self.data.shape[0] * 2],
            ),
            axis=1,
        )
    def plot_en(self, slip=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.quiver(self.data.x *1e-3, self.data.y * 1e-3, self.data.E, self.data.N)
        if slip is not None:
            model = self.get_model(slip)
            ax.quiver(self.data.x *1e-3, self.data.y * 1e-3, model.E, model.N, color='r')
        return ax

  
