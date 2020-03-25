import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from psokada.okada_stress import okada_stress, okada_stress_thread


class Population:
    def __init__(self, x, y, z, dip, strike, rake, ds, dn, n_hat, s_hat):
        self.x = x
        self.y = y
        self.z = z
        self.dip = dip
        self.strike = strike
        self.rake = rake
        self.ds = ds
        self.dn = dn
        self.n_hat = n_hat
        self.s_hat = s_hat

class Seismisity:

    def __init__(self, df, mu=0.5, lambda_l=50e9, shear_m=30e9):
        self.df = pd.read_csv(df)
        self.mu = mu
        self.lambda_l = lambda_l
        self.shear_m = shear_m
        self.G_ss = None
        self.G_ds = None

        def normal(strike, dip):
            nz = np.cos(dip)
            nx = np.cos(strike) * np.sin(dip)
            ny = -np.sin(strike ) * np.sin(dip)
            return np.array([nx, ny, nz]).reshape(-1, 1)

        def shear_hat(strike, dip, rake):
            ss = np.pi / 2 - strike
            l_s = np.cos(rake)
            l_d = np.sin(rake)
            nz = -np.sin(dip) * l_d
            l = np.cos(dip) * l_d

            x_s = np.cos(ss) * l_s
            y_s = np.sin(ss) * l_s

            x_d = np.cos(strike) * l
            y_d = np.sin(strike) * l
            return np.array([x_s + x_d, y_s + y_d, nz]).reshape(-1, 1)

        self.populations = []
        for i, row in self.df.iterrows():
            self.populations.append(Population(row.x, row.y, row.z, row.dip, row.strike, row.rake, row.ds, row.ds, normal(row.strike, row.dip), shear_hat(row.strike, row.dip, row.rake)))
        self.n_hats = np.squeeze(np.stack([p.n_hat for p in self.populations]))
        self.s_hat = np.squeeze(np.stack([p.s_hat for p in self.populations]))



    def build_ker(self, strike_element, dip_element, plains):
        if strike_element == 0:
            self.G_ss = np.zeros((self.df.shape[0], 0))
        else:
            self.G_ss = self.build_ker_element(strike_element, 0, plains)
        if dip_element == 0:
            self.G_ds = np.zeros((self.df.shape[0], 0))
        else:
            self.G_ds = self.build_ker_element(0, dip_element, plains)

    def build_ker_element(self, strike_element, dip_element, plains):
        n = 0
        for p in plains:
            n += len(p.sources)
        G = np.zeros((self.df.shape[0], n))
        for i, row in self.df.iterrows():
            stress = np.zeros((3, 3))
            for plain in plains:
                for isr, sr in enumerate(plain.sources):
                    okada_stress(sr.e_t, sr.n_t, sr.depth_t, sr.ccw_to_x_stk, sr.dip, sr.length, sr.width,
                                 plain.strike_element * strike_element, plain.dip_element * dip_element, 0, row.x, row.y, row.z, stress,
                                 self.lambda_l, self.shear_m)
                    t = np.squeeze(stress.dot(self.populations[i].n_hat))
                    tn = np.squeeze(t.dot(self.populations[i].n_hat))
                    ts = np.squeeze(t.dot(self.populations[i].s_hat))
                    G[i, isr] = ts - self.mu * tn
        return G

    def compute_source_stress(self, sr, strike_element, dip_element):
        stress = okada_stress_thread(sr.e_t, sr.n_t, sr.depth_t, sr.ccw_to_x_stk, sr.dip, sr.length, sr.width,
                          strike_element,  dip_element, 0, self.df.x.values, self.df.y.values, self.df.z.values,
                         self.lambda_l, self.shear_m, self.df.shape[0])
        t = np.einsum('ijk,ik->ij', stress, self.n_hats)
        tn = np.einsum('ij,ij->i', t, self.n_hats)
        ts = np.einsum('ij,ij->i', t, self.s_hat)
        return ts - self.mu * tn

    def plot_stress(self, plains, cmap='seismic', ax=None, sol=None, vmin=None, vmax=None):
        plot_color_bar = False
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            plot_color_bar = True
        if sol is None:
            sol = self.df.ds
        for p in plains:
            p.plot_sources(None, ax, cmap)
        sc = ax.scatter(self.df.x, self.df.y, -self.df.z, c=sol, cmap=cmap, vmin=vmin, vmax=vmax)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        if plot_color_bar:
            plt.colorbar(sc)

    def plot_sol(self, plains, slip, cmap='seismic', vmin=None, vmax=None):
        fig, axs = plt.subplots(1, 3, subplot_kw={'projection':'3d'})
        G = self.get_G()
        b = G.dot(slip)
        self.plot_stress(plains, cmap, axs[0], vmin=vmin, vmax=vmax)
        self.plot_stress(plains, cmap, axs[1], b, vmin=vmin, vmax=vmax)
        self.plot_stress(plains, cmap, axs[2], self.df.ds - b, vmin=vmin, vmax=vmax)


    def calc_stress(self, strike_element, dip_element, plains):
        tn = []
        ts = []
        coulomb = []
        for i, row in self.df.iterrows():
            stress = np.zeros((3, 3))
            for plain in plains:
                for isr, sr in enumerate(plain.sources):
                    if (sr.strike_slip < 1e-7 and sr.dip_slip < 1e-7):
                        continue
                    stress += sr.stress(row.x, row.y, row.z, strike_element * plain.strike_element,
                                        dip_element * plain.dip_element, self.lambda_l, self.shear_m)
            t = np.squeeze(stress.dot(self.populations[i].n_hat))
            tn.append(np.squeeze(t.dot(self.populations[i].n_hat)))
            ts.append(np.squeeze(t.dot(self.populations[i].s_hat)))
            coulomb.append(ts[-1] - self.mu * tn[-1])
        return np.array(ts), np.array(tn), np.array(coulomb)

    def get_G(self):
        return np.concatenate((self.G_ss, self.G_ds), axis=1)
    def get_b(self):
        return self.df.ds

    def resample_model(self, source_plain, source_ind, mat_ind, strike_element, dip_element, plains):
        plain = plains[source_plain]
        sr = plain.sources[source_ind]
        SR = sr.make_new_source()
        if strike_element != 0:
            B_strike = np.delete(self.G_ss, mat_ind, axis=1)
            for new_sr, k in zip(SR, [mat_ind, mat_ind + 1, mat_ind + 2, mat_ind + 3]):
                B_strike = np.insert(B_strike, k,
                                     self.compute_source_stress(new_sr, plain.strike_element * strike_element, 0), axis=1)
        else:
            B_strike = self.G_ss
        if dip_element != 0:
            B_dip = np.delete(self.G_ds, mat_ind, axis=1)
            for new_sr, k in zip(SR, [mat_ind, mat_ind + 1, mat_ind + 2, mat_ind + 3]):
                B_dip = np.insert(B_dip, k, self.compute_source_stress(new_sr, 0, plain.dip_element*dip_element),
                                  axis=1)
        else:
            B_dip = self.G_ds
        G = np.concatenate((B_strike, B_dip), axis=1)
        return G

    def insert_column(self, strike_element, dip_element, mat_ind, SR):
        if strike_element != 0:
            B_strike = np.delete(self.G_ss, mat_ind, axis=1)
            for new_sr, k in zip(SR, [mat_ind, mat_ind + 1, mat_ind + 2, mat_ind + 3]):
                B_strike = np.insert(B_strike, k,
                                     self.compute_source_stress(new_sr, strike_element, 0), axis=1)
            self.G_ss = B_strike
        if dip_element != 0:
            B_dip = np.delete(self.G_ds, mat_ind, axis=1)
            for new_sr, k in zip(SR, [mat_ind, mat_ind + 1, mat_ind + 2, mat_ind + 3]):
                B_dip = np.insert(B_dip, k, self.compute_source_stress(new_sr, 0, dip_element), axis=1)
            self.G_ds = B_dip

