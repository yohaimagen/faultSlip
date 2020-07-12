from inversion import Inversion
import numpy as np
import matplotlib.pyplot as plt
from faultSlip.point_source.point_source import py_point_source_strain
from scipy.spatial import cKDTree
from scipy.optimize import brute
import multiprocessing
import itertools
import pickle




class Genrate_catalog:
    def __init__(self, catalog, Gf, kdtree, slip_Gf, T, c, p, k, m_min, sub_dip=0.1, sub_strike=0.1):
        self.catalog = catalog
        if self.catalog is not None:
            self.catalog = np.load(self.catalog)
        self.Gf = Gf
        if self.Gf is not None:
            self.Gf = np.load(self.Gf)
        self.slip_Gf = slip_Gf
        if self.slip_Gf is not None:
            self.slip_Gf = np.load(self.slip_Gf)
        self.kdtree = kdtree
        if self.kdtree is not None:
            with open(self.kdtree, 'rb') as f:
                self.kdtree = pickle.load(f)
        self.sub_dip = sub_dip
        self.sub_strike = sub_strike
        self.T = T
        self.c = c
        self.p = p
        self.k = k
        self.m_min = m_min


    def calc_Gf(self, inv: Inversion, path=None):
        x, y, z = inv.plains[0].get_mesh(self.sub_strike, self.sub_dip)
        X = np.stack((x.flatten(), y.flatten(), z.flatten())).T
        Gf = []
        for i in range(self.catalog.shape[0]):
            gf = self.calc_strain_ps(X, self.catalog[i, 4], self.catalog[i, 0], self.catalog[i, 1], self.catalog[i, 2])
            gf[gf < 0.0] = 0.0
            gf = gf / np.sum(gf)
            Gf.append(gf)
        self.Gf = np.stack(Gf)
        self.kdtree = cKDTree(X)
        if path is not None:
            np.save(path + '.npy', self.Gf)
            with open(path + '_kdtree.pickel', 'wb') as f:
                pickle.dump(self.kdtree, f)


    def as_contribution(self, inv: Inversion):

        expected_as = []
        NT = ((self.T - self.catalog[:, 3] + self.c) ** (1 - self.p) - self.c ** (1 - self.p)) / (1 - self.p)
        N = (self.k * np.power(10, self.catalog[:, 4] - self.m_min)) * NT
        for p in inv.plains:
            for s in p.sources:
                inds = self.get_Gf_inds(s)
                gf = np.sum(self.Gf[:, inds], axis=1)
                expected_as.append(np.sum(gf * N))
        return np.array(expected_as)

    def get_Gf_inds(self, s):
        corners = s.get_corners()
        if s.length == s.width:
            mid = np.mean(corners, axis=0)
            inds = self.kdtree.query_ball_point(mid, s.length / 2.0, np.inf)
        else:
            l, w = s.length, s.width
            if w > l:
                l, w = w, l
            if l - w > w / 2.0:
                raise Exception('ratio between dislocation length and width more than 1.5')
            else:
                sdeast = np.cos(s.ccw_to_x_stk) * w * 0.5
                sdnorth = np.sin(s.ccw_to_x_stk) * w * 0.5
                ddeast = np.cos(s.ccw_to_x_dip) * np.cos(s.dip) * w * 0.5
                ddnorth = np.sin(s.ccw_to_x_dip) * np.cos(s.dip) * w * 0.5
                ddz = np.sin(s.dip) * w * 0.5
                p = np.zeros((4, 3))
                p[0, 0] = corners[0, 0] + sdeast + ddeast
                p[0, 1] = corners[0, 1] + sdnorth + ddnorth
                p[0, 2] = corners[0, 2] + ddz

                p[1, 0] = corners[1, 0] - sdeast + ddeast
                p[1, 1] = corners[1, 1] - sdnorth + ddnorth
                p[1, 2] = corners[1, 2] + ddz

                p[2, 0] = corners[2, 0] - sdeast + ddeast
                p[2, 1] = corners[2, 1] - sdnorth + ddnorth
                p[2, 2] = corners[2, 2] - ddz

                p[3, 0] = corners[3, 0] + sdeast + ddeast
                p[3, 1] = corners[3, 1] + sdnorth + ddnorth
                p[3, 2] = corners[3, 2] - ddz
                inds = self.kdtree.query_ball_point(p, s.length / 2.0, np.inf)
                inds = np.unique(np.concatenate(inds))
        return inds
    def calc_kernal(self, inv: Inversion, shear_m = 30e9):
        ker = np.zeros((inv.get_sources_num(), inv.get_sources_num()))
        omori = ((self.T + self.c) ** (1 - self.p) - self.c ** (1 - self.p)) / (1 - self.p)
        inds = []
        for p in inv.plains:
            for s in p.sources:
                inds.append(self.get_Gf_inds(s))
        i = 0
        for p in inv.plains:
            for s in p.sources:
                gr = (((shear_m * s.length * s.width * 1e6) ** (2.0 / 3)) / (10 ** (9.05 / 1.5 + self.m_min))) * self.k
                for j in range(inv.get_sources_num()):
                        ker[i, j] = (np.sum(self.slip_Gf[i, inds[j]]) * gr * omori)
                i += 1
        return (ker) ** (1.5)


    def as_in_disloc(self, inv: Inversion, min_dis=3.0):
        disloc_corners = np.zeros((0, 4, 3))
        for p in inv.plains:
            for s in p.sources:
                disloc_corners = np.concatenate((disloc_corners, s.get_corners().reshape(1, 4, 3)), axis=0)
        dists = self.distance(disloc_corners[:,  0], disloc_corners[:,  1], disloc_corners[:,  2], disloc_corners[:,  3], self.catalog[:, 0:3])
        mins = np.min(dists, axis=0)
        mask = mins < min_dis
        dists = dists[:, mask]
        mins = mins[mask]
        dists = dists - mins.reshape(1, -1)
        cum_as = np.sum(dists < 1e-20, axis=1)
        return cum_as

    def distance(self, t1, t2, t3, t4, p):


        '''
        compute the distance betwen aftershck catalog with n aftershocks to fault dislocations with k dislocations
        t1 ... t4 corners of the dislocations where ti is from the shape (k, 3) and orderd counteclock wise
        p are the aftershocks locations of the shape (n, 3)

        '''
        v = t2 - t1
        u = t4 - t1
        n = np.cross(v, u)
        n = n / np.linalg.norm(n, axis=1).reshape(-1, 1)
        d = - np.einsum('ij,ij->i', n, t1)

        D = n.dot(p.T) + d.reshape(-1, 1)

        pp = p - np.einsum('ij,ik->ikj', n, D)

        v = v / np.linalg.norm(v, axis=1).reshape(-1, 1)
        u = u / np.linalg.norm(u, axis=1).reshape(-1, 1)

        t1 = np.stack((np.einsum('ij,ij->i', v, t1), np.einsum('ij,ij->i', u, t1))).T
        t2 = np.stack((np.einsum('ij,ij->i', v, t2), np.einsum('ij,ij->i', u, t2))).T
        t3 = np.stack((np.einsum('ij,ij->i', v, t3), np.einsum('ij,ij->i', u, t3))).T
        t4 = np.stack((np.einsum('ij,ij->i', v, t4), np.einsum('ij,ij->i', u, t4))).T
        pp = np.stack((np.einsum('ij,ikj->ki', v, pp), np.einsum('ij,ikj->ki', u, pp))).T

        T = np.stack([t1, t2, t3, t4], axis=1)

        x_max = np.max(T[:, :, 0], axis=1).reshape(-1, 1)
        x_min = np.min(T[:, :, 0], axis=1).reshape(-1, 1)
        y_max = np.max(T[:, :, 1], axis=1).reshape(-1, 1)
        y_min = np.min(T[:, :, 1], axis=1).reshape(-1, 1)

        D_plain = np.zeros((pp.shape[0], pp.shape[1]))

        mask_x = np.logical_and(pp[:, :, 0] <= x_max, pp[:, :, 0] >= x_min)
        mask_y = np.logical_and(pp[:, :, 1] <= y_max, pp[:, :, 1] >= y_min)

        mask_corners = np.logical_not(np.logical_and(mask_x, mask_y))

        mask = np.logical_and(mask_x, pp[:, :, 1] > y_max)
        D_plain[mask] = pp[mask, 1] - np.repeat(y_max, pp.shape[1], axis=1)[mask]
        mask_corners = np.logical_and(mask_corners, np.logical_not(mask))

        mask = np.logical_and(mask_x, pp[:, :, 1] < y_min)
        D_plain[mask] = np.repeat(y_min, pp.shape[1], axis=1)[mask] - pp[mask, 1]
        mask_corners = np.logical_and(mask_corners, np.logical_not(mask))

        mask = np.logical_and(pp[:, :, 0] > x_max, mask_y)
        D_plain[mask] = pp[mask, 0] - np.repeat(x_max, pp.shape[1], axis=1)[mask]
        mask_corners = np.logical_and(mask_corners, np.logical_not(mask))

        mask = np.logical_and(pp[:, :, 0] < x_min, mask_y)
        D_plain[mask] = np.repeat(x_min, pp.shape[1], axis=1)[mask] - pp[mask, 0]
        mask_corners = np.logical_and(mask_corners, np.logical_not(mask))

        D_plain[mask_corners] = np.min(np.linalg.norm(
            np.repeat(T.reshape(-1, 1, 4, 2), pp.shape[1], axis=1)[mask_corners] - pp[mask_corners].reshape(-1, 1, 2),
            axis=2), axis=1)

        return np.sqrt(D ** 2 + D_plain ** 2)

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

    def calc_slip_Gf(self, inv: Inversion, path=None):
        x, y, z = inv.plains[0].get_mesh(self.sub_strike, self.sub_dip)
        X = np.stack((x.flatten(), y.flatten(), z.flatten())).T
        lambda_l = 50e9
        shear_m = 30e9
        n_hat = np.array([0, -1, 0], dtype=np.float)
        s_hat = np.array([1, 0, 0], dtype=np.float)
        n = np.sum([len(p.sources) for p in inv.plains])
        strain = np.zeros((n, X.shape[0], 3, 3))
        i = 0
        for plain in inv.plains:
            for isr, sr in enumerate(plain.sources):
                strain[i] += sr.strain_thread_gf(X[:, 0], X[:, 1], X[:, 2], inv.strike_element * plain.strike_element,
                                           inv.dip_element * plain.dip_element, lambda_l, shear_m)
                i += 1
        t = np.einsum('lijk,k -> lij', strain, n_hat)
        tn = np.einsum('lij,j->li', t, n_hat)
        ts = np.einsum('lij,j->li', t, s_hat)
        coulomb = ts - 0.6 * tn
        max_strain = (10e6 +0.8 * 23e3 * X[:, 2] * 1e3) / lambda_l
        max_strain = np.repeat(max_strain.reshape(1, -1), coulomb.shape[0], axis=0)
        mask = coulomb > max_strain
        coulomb[mask] = max_strain[mask]
        coulomb[coulomb < 0.0] = 0.0
        coulomb = coulomb / np.sum(coulomb, axis=1).reshape(-1, 1)
        if path is not None:
            np.save(path, coulomb)
        self.slip_Gf = coulomb

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

    def generate_catalog_slip(self, inv: Inversion, T, X, kdtree, k=2.84e-3, p=1.07, c=1.78e-5, m_min=2.5):
        mw = inv.moment_magnitude()
        print(mw)
        inv.assign_slip()
        NT = ((T + c) ** (1 - p) - c ** (1 - p)) / (1 - p)
        N = int(((k * np.power(10, mw - m_min)) * NT))
        if N < 1:
            return np.zeros((0, 5))
        else:
            Gf = self.calc_strain(X, inv)
            Gf[Gf < 0] = 0.0
            NS = np.sum(Gf) * 1e5
            Gf = Gf / NS
            x = inv.plains[0].sample_points(5000)
            t = np.random.uniform(0, T, 5000)
            inds = kdtree.query(x)[1]
            pdf_0 = Gf[inds] * (((t + c) ** -p) / NT)
            p_max = np.max(pdf_0)
            p_min = 0#pdf_0[1000]
            # plt.plot(np.sort(pdf_0))
            # plt.show()

            n_samp = 10000
            ntrail = 0
            accepted = np.zeros((0, 4))

            while accepted.shape[0] < N:
                if ntrail %10 == 0:
                    print(f'{ntrail}, {accepted.shape[0]}')
                x = inv.plains[0].sample_points(n_samp)
                t = np.random.uniform(0, T, n_samp)
                inds = kdtree.query(x)[1]
                pdf = Gf[inds] * (((t + c) ** -p) / NT)
                y = np.random.uniform(p_min, p_max, n_samp)
                mask = y < pdf
                accepted = np.concatenate((accepted, np.concatenate((x[mask], t[mask].reshape(-1, 1)), axis=1)), axis=0)
                if accepted.shape[0] > N:
                    accepted = accepted[:N]
                ntrail += 1
            accepted = np.concatenate((accepted, self.sample_magnitudes(mw, m_min, N).reshape(-1, 1)), axis=1)
            return accepted

    def generate_catalog_ps(self, inv: Inversion, mw, x, y, z, t_origin, T_t, X, kdtree, k=2.84e-3, p=1.07, c=1.78e-5, m_min=2.5):
        T = T_t - t_origin
        if T <= 0:
            return np.zeros((0, 5))
        NT = ((T + c) ** (1 - p) - c ** (1 - p)) / (1 - p)
        N = int(np.rint((k * np.power(10, mw - m_min)) * NT))
        if N < 1:
            return np.zeros((0, 5))
        else:
            Gf = self.calc_strain_ps(X, mw, x, y, z)
            Gf[Gf < 0] = 0.0
            NS = np.sum(Gf) * 1e5
            Gf = Gf / NS
            x = inv.plains[0].sample_points(5000)
            t = np.random.uniform(0, T, 5000)
            inds = kdtree.query(x)[1]
            pdf_0 = Gf[inds] * (((t + c) ** -p) / NT)
            p_max = np.nanmax(pdf_0)
            p_min = 0

            ntrail = 0
            accepted = np.zeros((0, 4))

            n_per_round = 1000
            while accepted.shape[0] < N:
                x = inv.plains[0].sample_points(n_per_round)
                t = np.random.uniform(0, T, n_per_round)
                inds = kdtree.query(x)[1]
                pdf = Gf[inds] * (((t + c) ** -p) / NT)
                g = np.random.uniform(p_min, p_max, n_per_round)
                mask = g < pdf
                accepted = np.concatenate((accepted, np.concatenate((x[mask], t[mask].reshape(-1, 1)), axis=1)), axis=0)
                if accepted.shape[0] > N:
                    accepted = accepted[:N]
                ntrail += 1
            accepted = np.concatenate((accepted, self.sample_magnitudes(mw, m_min, N).reshape(-1, 1)), axis=1)
            accepted[:, 3] += t_origin
            for i in range(accepted.shape[0]):
                accepted = np.concatenate((accepted, self.generate_catalog_ps(inv, accepted[i, 4], accepted[i, 0],
                                                                                accepted[i, 1], accepted[i, 2],
                                                                                accepted[i, 3], T_t, X, kdtree, k, p, c, m_min)))
            return accepted

    def generate_catalog(self, inv: Inversion, T, k=2.84e-3, p=1.07, c=1.78e-5, m_min=2.5, sub_dip=0.1, sub_strike=0.1):

        x, y, z = inv.plains[0].get_mesh(sub_dip, sub_strike)
        X = np.stack((x.flatten(), y.flatten(), z.flatten())).T
        kdtree = cKDTree(X)

        catalog = self.generate_catalog_slip(inv, T, X, kdtree, k, p, c, m_min)
        as_catalog = np.zeros((0, 5))
        for i in range(catalog.shape[0]):
            if i % 10 == 0:
                print(i)
            as_catalog = np.concatenate((as_catalog, self.generate_catalog_ps(inv, catalog[i, 4], catalog[i, 0],
                                                                                catalog[i, 1], catalog[i, 2],
                                                                                catalog[i, 3], T, X, kdtree, k, p, c, m_min)))
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
        inv.assign_slip()
        x, y, z = inv.plains[0].get_mesh(0.1, 0.1)
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
