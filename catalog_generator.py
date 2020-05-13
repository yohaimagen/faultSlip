from inversion import Inversion
import numpy as np
import matplotlib.pyplot as plt
from faultSlip.point_source.point_source import py_point_source_strain
from scipy.spatial import KDTree
from scipy.optimize import brute
import multiprocessing
import itertools



class Genrate_catalog:
    def __init__(self):
        pass

    def calc_strain(self, X, inv: Inversion):
        lambda_l = 50e9
        shear_m = 30e9
        n_hat = np.array([0, -1, 0], dtype=np.float)
        s_hat = np.array([1, 0, 0], dtype=np.float)
        strain = np.zeros((X.shape[0], 3, 3))
        for plain in inv.plains:
            for isr, sr in enumerate(plain.sources):
                if (sr.strike_slip < 1e-7 and sr.dip_slip < 1e-7):
                    continue
                strain += sr.strain_thread(X[:, 0], X[:, 1], X[:, 2], inv.strike_element * plain.strike_element,
                                           inv.dip_element * plain.dip_element, lambda_l, shear_m)
        t = np.einsum('ijk,k -> ij', strain, n_hat)
        tn = t.dot(n_hat)
        ts = t.dot(s_hat)
        coulomb = ts - 0.6 * tn
        max_strain = (10e6 +0.8 * 23e3 * X[:, 2] * 1e3) / lambda_l
        mask = coulomb > max_strain
        coulomb[mask] = max_strain[mask]
        return coulomb

    def calc_strain_ps(self, X, mw, x, y, z):
        m0 = 10 ** (1.5 * mw + 9.05)
        lambda_l = 50e9
        shear_m = 30e9
        n_hat = np.array([0, -1, 0], dtype=np.float)
        s_hat = np.array([1, 0, 0], dtype=np.float)
        # pool = ThreadPool(12)
        # def my_func(A):
        #     a, b, c = A
        #     return py_point_source_strain(x, y, z, np.pi / 2.0, np.pi / 2.0, m0, 0.0, 0.0, 0.0, a, b,
        #                                        c, lambda_l, shear_m)
        # stress = pool.map(my_func, zip(X[:, 0], X[:, 1], X[:, 2]))
        # pool.close()
        # pool.join()
        # stress = np.stack(stress)
        stress = np.zeros((X.shape[0], 3, 3))
        for i in range(X.shape[0]):
            stress[i] = py_point_source_strain(x, y, z, np.pi / 2.0, np.pi / 2.0, m0, 0.0, 0.0, 0.0, X[i, 0], X[i, 1],
                                               X[i, 2], lambda_l, shear_m)

        t = np.einsum('ijk,k -> ij', stress, n_hat)
        tn = t.dot(n_hat)
        ts = t.dot(s_hat)
        coulomb = ts - 0.6 * tn
        max_strain = (10e6 + 0.8 * 23e3 * X[:, 2] * 1e3) / lambda_l
        mask = coulomb > max_strain
        coulomb[mask] = max_strain[mask]
        return coulomb

    def sample_magnitudes(self, mw, m_min, N, b_l=0.5, a=4.375, b=0.746):
        def gr(a, b, Mw):
            return 10**(a - b*Mw)

        p_max = gr(a, b, m_min)
        p_min = 0.0
        naccept = 0
        ntrail = 0
        accepted = np.zeros(0)
        while naccept < N:
            m = np.random.uniform(m_min, mw, N - naccept)
            pdf = gr(mw - b_l, 1.0, m)
            g = np.random.uniform(p_min, p_max, N - naccept)
            mask = g < pdf
            accepted = np.concatenate((accepted, m[mask]))
            naccept += np.sum(mask)
            ntrail += 1
        return accepted

    def generate_catalog_slip(self, inv: Inversion, T, k=2.84e-3, p=1.14, c=0.0016709, m_min=2.5):
        mw = inv.moment_magnitude()
        print(mw)
        inv.assign_slip()
        N = int(((k * np.power(10, mw - m_min)) / (1 - p) * ((T + c) ** (1 - p) - c ** (1 - p))))
        if N < 1:
            return np.zeros((0, 6))
        else:
            X = inv.plains[0].sample_points(10000)
            t = np.random.uniform(0, T, 10000)
            pdf_0 = self.calc_strain(X, inv) * ((t + c) ** -p)
            pdf_0 = np.sort(pdf_0)
            p_max = pdf_0[-1]
            p_min = 0#pdf_0[1000]
            # plt.plot(np.sort(pdf_0))
            # plt.show()

            n_samp = 10000
            ntrail = 0
            accepted = np.zeros((0, 5))

            while accepted.shape[0] < N:
                if ntrail %10 == 0:
                    print(f'{ntrail}, {accepted.shape[0]}')
                X = inv.plains[0].sample_points((n_samp))
                t = np.random.uniform(0, T, n_samp)
                pdf = self.calc_strain(X, inv) * ((t + c) ** -p)
                y = np.random.uniform(p_min, p_max, n_samp)
                mask = y < pdf
                accepted = np.concatenate((accepted, np.concatenate((X[mask], t[mask].reshape(-1, 1), pdf[mask].reshape(-1, 1)), axis=1)), axis=0)
                if accepted.shape[0] > N:
                    accepted = accepted[:N]
                ntrail += 1
            accepted = np.concatenate((accepted, self.sample_magnitudes(mw, m_min, N).reshape(-1, 1)), axis=1)
            return accepted

    def generate_catalog_ps(self, inv: Inversion, mw, x, y, z, t_origin, T_t, k=2.84e-3, p=1.07, c=1.78e-5, m_min=2.5):
        T = T_t - t_origin
        if T <= 0:
            return np.zeros((0, 6))
        N = int(np.rint((k * np.power(10, mw - m_min)) / (1 - p) * ((T + c) ** (1 - p) - c ** (1 - p))))
        if N < 1:
            return np.zeros((0, 6))
        else:
            X = inv.plains[0].sample_points(10000)
            t = np.random.uniform(0, T, 10000)
            pdf_0 = self.calc_strain_ps(X, mw, x, y, z) * ((t + c) ** -p)
            p_max = np.nanmax(pdf_0)
            p_min = 0

            ntrail = 0
            accepted = np.zeros((0, 5))

            n_per_round = 1000
            while accepted.shape[0] < N:
                X = inv.plains[0].sample_points(n_per_round)
                t = np.random.uniform(0, T, n_per_round)
                pdf = self.calc_strain_ps(X, mw, x, y, z) * ((t + c) ** -p)
                g = np.random.uniform(p_min, p_max, n_per_round)
                mask = g < pdf
                accepted = np.concatenate((accepted, np.concatenate((X[mask], t[mask].reshape(-1, 1), pdf[mask].reshape(-1, 1)), axis=1)), axis=0)
                if accepted.shape[0] > N:
                    accepted = accepted[:N]
                ntrail += 1
            accepted = np.concatenate((accepted, self.sample_magnitudes(mw, m_min, N).reshape(-1, 1)), axis=1)
            accepted[:, 3] += t_origin
            for i in range(accepted.shape[0]):
                accepted = np.concatenate((accepted, self.generate_catalog_ps(inv, accepted[i, 5], accepted[i, 0],
                                                                                accepted[i, 1], accepted[i, 2],
                                                                                accepted[i, 3], T, k, p, c, m_min)))
            return accepted

    def generate_catalog(self, inv: Inversion, T, k=2.84e-3, p=1.07, c=1.78e-5, m_min=2.5):
        catalog = self.generate_catalog_slip(inv, T, k, p, c, m_min)
        as_catalog = np.zeros((0, 6))
        for i in range(catalog.shape[0]):
            as_catalog = np.concatenate((as_catalog, self.generate_catalog_ps(inv, catalog[i, 5], catalog[i, 0],
                                                                                catalog[i, 1], catalog[i, 2],
                                                                                catalog[i, 3], T, k, p, c, m_min)))
        return np.concatenate((catalog, as_catalog), axis=0)


    def fit_catalog(self, cat, inv: Inversion, save_gf=None, load_gf=None, ranges=((-20, -1), (-25, -2), (1.000001, 1.5)),  Ns = (10, 10, 10)):
        cat = cat[np.argsort(cat[:, 3])]
        sub_dip =  int(np.rint(inv.plains[0].total_width * 10.0))
        sub_strike = int(np.rint(inv.plains[0].plain_length * 10.0))
        inv.assign_slip()
        x, y, z = inv.plains[0].get_mesh(sub_dip, sub_strike)
        X = np.stack((x.flatten(), y.flatten(), z.flatten())).T
        if load_gf is None:
            Gf = []
            Gf.append(self.calc_strain(X, inv))
            for i in range(cat.shape[0]):
                Gf.append(self.calc_strain_ps(X, cat[i, 5], cat[i, 0], cat[i, 1], cat[i, 2]))
            Gf = np.stack(Gf)
        else:
            Gf = np.load(load_gf)
        if save_gf is not None:
            np.save(save_gf, Gf)
        Gf = Gf / (np.sum(Gf, axis=1).reshape(-1, 1) * 0.01)
        T = np.max(cat[:, 3])


        kdtree = KDTree(X)



        sranges = list((np.linspace(a[0], a[1], n) for a, n in zip(ranges, Ns)))

        grid = np.stack(np.meshgrid(*sranges))
        args = []
        # for k in ranges(Ns):
        #     for l in range(Ns):
        #         for m in range(Ns):
        #             args .append((grid[0, k, l, m], sranges[1], sranges[2], [Gf], [cat], [kdtree], [T]))

        args = list(itertools.product(sranges[0], sranges[1], sranges[2], [Gf], [cat], [kdtree], [T], [2.5]))
        inds = np.linspace(0, len(args), 12, dtype=np.int)
        pool_arrgs = [(args[inds[i]: inds[i+1]]) for i in range(inds.shape[0] - 1)]
        p = multiprocessing.Pool(12)
        results = p.map(f, pool_arrgs)
        p.close()
        p.join()
        jout = np.array(list(itertools.chain(*results))).reshape(sranges[0].shape[0], sranges[1].shape[0], sranges[2].shape[0])

        # x0 , fval, grid, jout = brute(logliklihod, ranges, args=(Gf, cat, kdtree, T), Ns=10, full_output=True, workers=12)
        # np.save('./x0.npy', x0)
        # np.save('./fval.npy', np.array([fval]))
        np.save('./grid.npy', grid)
        np.save('./jout.npy', jout)

    def mc_fit_catalog(self, cat, inv: Inversion, save_gf=None, load_gf=None, ranges=((-20, -1), (-25, -2), (1.000001, 1.5)), samples=1000):
        cat = cat[np.argsort(cat[:, 3])]
        sub_dip = int(np.rint(inv.plains[0].total_width * 10.0))
        sub_strike = int(np.rint(inv.plains[0].plain_length * 10.0))
        inv.assign_slip()
        x, y, z = inv.plains[0].get_mesh(sub_dip, sub_strike)
        X = np.stack((x.flatten(), y.flatten(), z.flatten())).T
        if load_gf is None:
            Gf = []
            Gf.append(self.calc_strain(X, inv))
            for i in range(cat.shape[0]):
                Gf.append(self.calc_strain_ps(X, cat[i, 5], cat[i, 0], cat[i, 1], cat[i, 2]))
            Gf = np.stack(Gf)
        else:
            Gf = np.load(load_gf)
        if save_gf is not None:
            np.save(save_gf, Gf)
        Gf = Gf / (np.sum(Gf, axis=1).reshape(-1, 1) * 0.01)
        T = np.max(cat[:, 3])

        kdtree = KDTree(X)

        sranges = list((np.random.uniform(a[0], a[1], samples) for a in ranges))
        grid = np.stack(sranges)
        args = []
        for k in range(samples):
            args.append((sranges[0][k], sranges[1][k], sranges[2][k], Gf, cat, kdtree, T, 2.5))

        inds = np.linspace(0, len(args), 12, dtype=np.int)
        pool_arrgs = [(args[inds[i]: inds[i + 1]]) for i in range(inds.shape[0] - 1)]
        p = multiprocessing.Pool(12)
        results = p.map(f, pool_arrgs)
        p.close()
        p.join()
        jout = np.array(list(itertools.chain(*results)))

        # x0 , fval, grid, jout = brute(logliklihod, ranges, args=(Gf, cat, kdtree, T), Ns=10, full_output=True, workers=12)
        # np.save('./x0.npy', x0)
        # np.save('./fval.npy', np.array([fval]))
        np.save('./mc_grid.npy', grid)
        np.save('./mc_jout.npy', jout)

        return results

def omori(t, ti, c, p):
    return (t - ti + c) ** (-p)

def Omori(t, c, p):
    return ((t + c) ** (-p + 1)) / (-p + 1) - ((c ** (-p + 1)) / (-p + 1))

def logliklihod(k, c, p, Gf, cat, kdtree, T, m_min):
    print(f'{k}, {c}, {p}')
    k = 10 ** k
    c = 10 ** c
    ll = 0
    LL = 0
    LL += np.power(10, 6.5 - m_min) * Omori(T, c, p) * np.sum(Gf[0]) * 0.01
    for i in range(cat.shape[0]):
        ll_i = 0
        ind = kdtree.query(cat[i, 0:3])[1]
        ll_i += k * np.power(10, 6.5 - m_min) * omori(cat[i, 3], 0, c, p) * Gf[0, ind]
        LL += np.power(10, cat[i, 5] - m_min) * Omori(T - cat[i, 3], c, p) * np.sum(Gf[i + 1]) * 0.01
        for j in range(1, i + 1):
            ll_i += k * np.power(10, cat[j, 5] - m_min) * omori(cat[i, 3], cat[j, 3], c, p) * Gf[j, ind]
        ll += np.log(ll_i)
    return ll - k * LL

def f(args):
    return [logliklihod(*arg) for arg in args]
