from copy import deepcopy

import numpy as np

from faultSlip.point_source.point_source import py_disp_point_source

# from okada_wrapper import dc3dwrapper
from faultSlip.psokada.okada_stress import *


# def okada_stress():
#     print 'change imports in souces.py'
class Point_sources:
    def __init__(self, point_sources):
        self.point_sources = []

        def key(elem):
            return int(elem.split("e")[-1])

        for key in sorted(point_sources, key=key):
            self.point_sources.append(Point_source(**point_sources[key]))

    def build_ker(self, x, y, z):
        Gs = [ps.build_ker(x, y, z) for ps in self.point_sources]
        return np.concatenate(Gs, axis=1)


class Point_source:
    def __init__(
        self, strike, dip, ss, ds, open, inflation, cords, mu=30e9, l_lambda=50e9
    ):
        self.strike = np.deg2rad(strike)
        self.dip = np.deg2rad(dip)
        self.ss = ss
        self.ds = ds
        self.open = open
        self.inflation = inflation
        self.east = cords[0]
        self.north = cords[1]
        self.depth = cords[2]
        self.mu = mu
        self.l_lambda = l_lambda
        self.name = 1

    def build_ker(self, x, y, z):
        if self.ss != 0:
            ess = np.zeros((x.shape[0], 1))
            nss = np.zeros((x.shape[0], 1))
            zss = np.zeros((x.shape[0], 1))
            for i in range(x.shape[0]):
                disp = py_disp_point_source(
                    self.east,
                    self.north,
                    self.depth,
                    self.strike,
                    self.dip,
                    1e18 * self.ss,
                    0,
                    0,
                    0,
                    x[i],
                    y[i],
                    z[i],
                    self.mu,
                    self.l_lambda,
                )
                ess[i, 0] = disp[0]
                nss[i, 0] = disp[1]
                zss[i, 0] = disp[2]
        else:
            ess = np.zeros((x.shape[0], 0))
            nss = np.zeros((x.shape[0], 0))
            zss = np.zeros((x.shape[0], 0))
        if self.ds != 0:
            eds = np.zeros((x.shape[0], 1))
            nds = np.zeros((x.shape[0], 1))
            zds = np.zeros((x.shape[0], 1))
            for i in range(x.shape[0]):
                disp = py_disp_point_source(
                    self.east,
                    self.north,
                    self.depth,
                    self.strike,
                    self.dip,
                    0,
                    self.ds * 1e18,
                    0,
                    0,
                    x[i],
                    y[i],
                    z[i],
                    self.mu,
                    self.l_lambda,
                )
                eds[i, 0] = disp[0]
                nds[i, 0] = disp[1]
                zds[i, 0] = disp[2]
        else:
            eds = np.zeros((x.shape[0], 0))
            nds = np.zeros((x.shape[0], 0))
            zds = np.zeros((x.shape[0], 0))
        if self.open != 0:
            eopen = np.zeros((x.shape[0], 1))
            nopen = np.zeros((x.shape[0], 1))
            zopen = np.zeros((x.shape[0], 1))
            for i in range(x.shape[0]):
                disp = py_disp_point_source(
                    self.east,
                    self.north,
                    self.depth,
                    self.strike,
                    self.dip,
                    0,
                    0,
                    0,
                    self.open * 1e18,
                    x[i],
                    y[i],
                    z[i],
                    self.mu,
                    self.l_lambda,
                )
                eopen[i, 0] = disp[0]
                nopen[i, 0] = disp[1]
                zopen[i, 0] = disp[2]
        else:
            eopen = np.zeros((x.shape[0], 0))
            nopen = np.zeros((x.shape[0], 0))
            zopen = np.zeros((x.shape[0], 0))
        if self.inflation != 0:
            einf = np.zeros((x.shape[0], 1))
            ninf = np.zeros((x.shape[0], 1))
            zinf = np.zeros((x.shape[0], 1))
            for i in range(x.shape[0]):
                disp = py_disp_point_source(
                    self.east,
                    self.north,
                    self.depth,
                    self.strike,
                    self.dip,
                    0,
                    0,
                    self.inflation * 1e18,
                    0,
                    x[i],
                    y[i],
                    z[i],
                    self.mu,
                    self.l_lambda,
                )
                einf[i, 0] = disp[0]
                ninf[i, 0] = disp[1]
                zinf[i, 0] = disp[2]
        else:
            einf = np.zeros((x.shape[0], 0))
            ninf = np.zeros((x.shape[0], 0))
            zinf = np.zeros((x.shape[0], 0))
        E = np.concatenate((ess, eds, eopen, einf), axis=1)
        N = np.concatenate((nss, nds, nopen, ninf), axis=1)
        Z = np.concatenate((zss, zds, zopen, zinf), axis=1)
        G = np.concatenate((E, N, Z), axis=0)
        return G

    def __str__(self):
        return f""""point_source{self.name}": {{
    "strike": {np.rad2deg(self.strike)},
    "dip": {np.rad2deg(self.dip)},
    "ss": {self.ss},
    "ds": {self.ds},
    "open": {self.open},
    "inflation": {self.inflation},
    "cords": [{self.east}, {self.north}, {self.depth}]
}}"""
