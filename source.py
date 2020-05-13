import numpy as np
from copy import deepcopy
# from okada_wrapper import dc3dwrapper
from psokada.okada_stress import *

# def okada_stress():
#     print 'change imports in souces.py'
class Source:
    def __init__(self, E, N, strike, dip, length, width, ss, ds, ts, x=0, y=0, a=0, b=0, c=0, open=0.0, east=None, north=None, depth=None):
        self.strike = strike
        self.dip = dip
        self.length = length
        self.width = width
        self.ss = ss
        self.ds = ds
        self.ts = ts
        self.open = open
        self.ccw_to_x_stk = np.pi / 2-self.strike # the angle between the fault and the x axis cunter clock wise
        self.ccw_to_x_dip = -self.strike
        if east is not None and north is not None and depth is not None:
            self.e = east
            self.n = north
            self.depth = depth
            self.depth_t = self.depth - self.width * np.sin(self.dip)
            l = self.width * np.cos(self.dip)
            ccw_to_x_up_dip = self.ccw_to_x_dip + np.pi
            self.e_t = self.e + np.cos(ccw_to_x_up_dip) * l
            self.n_t = self.n + np.sin(ccw_to_x_up_dip) * l
            self.depth_m = self.depth - 0.5 * self.width * np.sin(self.dip)
            l = 0.5 * self.width * np.cos(self.dip)
            self.e_m = self.e + np.cos(ccw_to_x_up_dip) * l
            self.n_m = self.n + np.sin(ccw_to_x_up_dip) * l
        else:
            self.e = a + x*np.cos(self.ccw_to_x_stk) + np.cos(self.ccw_to_x_dip) * np.cos(dip)*y
            self.n = b + x*np.sin(self.ccw_to_x_stk) + np.sin(self.ccw_to_x_dip) * np.cos(dip) * y
            self.depth = c + np.sin(dip)*y
            self.e_t = a + x*np.cos(self.ccw_to_x_stk) + np.cos(self.ccw_to_x_dip) * np.cos(dip)*(y-self.width)
            self.n_t = b + x * np.sin(self.ccw_to_x_stk) + np.sin(self.ccw_to_x_dip) * np.cos(dip) * (y-self.width)
            self.depth_t = c + np.sin(dip) * (y-self.width)
            self.e_m = a + x * np.cos(self.ccw_to_x_stk) + np.cos(self.ccw_to_x_dip) * np.cos(dip) * (y - 0.5 * self.width)
            self.n_m = b + x * np.sin(self.ccw_to_x_stk) + np.sin(self.ccw_to_x_dip) * np.cos(dip) * (y - 0.5 * self.width)
            self.depth_m = c + np.sin(dip) * (y - 0.5 * self.width)
        deast = np.cos(self.ccw_to_x_stk) * self.width * 0.5
        dnorth = np.sin(self.ccw_to_x_stk) * self.width * 0.5
        self.p1 = np.array([self.e_t - deast, self.n_t - dnorth, self.depth_t])
        self.p2 = np.array([self.e_t + deast, self.n_t + dnorth, self.depth_t])
        self.p3 = np.array([self.e + deast, self.n + dnorth, self.depth])
        self.p4 = np.array([self.e - deast, self.n - dnorth, self.depth])
        self.x = x
        self.y = y
        self.strike_slip = None
        self.dip_slip = None
        self.d_slip = None
        if E is not None:
            self.E = E-self.e
        if N is not None:
            self.N = N-self.n

    def make_new_source(self):
        z = self.depth - np.sin(self.dip) * self.width
        l = np.cos(self.dip) * self.width
        e = self.e - (self.length / 2 * np.cos(self.ccw_to_x_stk) + np.cos(self.ccw_to_x_dip) * l)
        n = self.n - (self.length / 2 * np.sin(self.ccw_to_x_stk) + np.sin(self.ccw_to_x_dip) * l)
        sr1 = deepcopy(self)
        sr1.length /= 2.0
        sr1.width /= 2.0
        sr1.x -= sr1.length / 2.0
        sr1.y -= sr1.width
        sr1.depth = z + np.sin(sr1.dip) * sr1.width
        sr1.e = e + sr1.length / 2 * np.cos(self.ccw_to_x_stk) + np.cos(self.ccw_to_x_dip) * np.cos(sr1.dip)*sr1.width
        sr1.n = n + sr1.length / 2 * np.sin(self.ccw_to_x_stk) + np.sin(self.ccw_to_x_dip) * np.cos(sr1.dip)*sr1.width

        sr1.depth_t = sr1.depth - sr1.width * np.sin(sr1.dip)
        tl = sr1.width * np.cos(sr1.dip)
        ccw_to_x_up_dip = sr1.ccw_to_x_dip + np.pi
        sr1.e_t = sr1.e + np.cos(ccw_to_x_up_dip) * tl
        sr1.n_t = sr1.n + np.sin(ccw_to_x_up_dip) * tl
        sr1.depth_m = sr1.depth - 0.5 * sr1.width * np.sin(sr1.dip)
        tl = 0.5 * sr1.width * np.cos(sr1.dip)
        sr1.e_m = sr1.e + np.cos(ccw_to_x_up_dip) * tl
        sr1.n_m = sr1.n + np.sin(ccw_to_x_up_dip) * tl


        sr2 = deepcopy(sr1)
        sr2.x += sr1.length
        sr2.e = e + sr2.length * 1.5 * np.cos(self.ccw_to_x_stk) + np.cos(self.ccw_to_x_dip) * np.cos(sr2.dip) * sr2.width
        sr2.n = n + sr2.length * 1.5 * np.sin(self.ccw_to_x_stk) + np.sin(self.ccw_to_x_dip) * np.cos(sr2.dip) * sr2.width
        sr2.depth = z + np.sin(sr2.dip) * sr2.width

        sr2.depth_t = sr2.depth - sr2.width * np.sin(sr2.dip)
        tl = sr2.width * np.cos(sr2.dip)
        ccw_to_x_up_dip = sr2.ccw_to_x_dip + np.pi
        sr2.e_t = sr2.e + np.cos(ccw_to_x_up_dip) * tl
        sr2.n_t = sr2.n + np.sin(ccw_to_x_up_dip) * tl
        sr2.depth_m = sr2.depth - 0.5 * sr2.width * np.sin(sr2.dip)
        tl = 0.5 * sr2.width * np.cos(sr2.dip)
        sr2.e_m = sr2.e + np.cos(ccw_to_x_up_dip) * tl
        sr2.n_m = sr2.n + np.sin(ccw_to_x_up_dip) * tl

        sr3 = deepcopy(sr1)
        sr3.y += sr1.width
        sr3.e = e + sr3.length / 2 * np.cos(self.ccw_to_x_stk) + np.cos(self.ccw_to_x_dip) * np.cos(sr3.dip) * sr3.width * 2
        sr3.n = n + sr3.length / 2 * np.sin(self.ccw_to_x_stk) + np.sin(self.ccw_to_x_dip) * np.cos(sr3.dip) * sr3.width * 2
        sr3.depth = z + np.sin(sr3.dip) * sr3.width * 2

        sr3.depth_t = sr3.depth - sr3.width * np.sin(sr3.dip)
        tl = sr3.width * np.cos(sr3.dip)
        ccw_to_x_up_dip = sr3.ccw_to_x_dip + np.pi
        sr3.e_t = sr3.e + np.cos(ccw_to_x_up_dip) * tl
        sr3.n_t = sr3.n + np.sin(ccw_to_x_up_dip) * tl
        sr3.depth_m = sr3.depth - 0.5 * sr3.width * np.sin(sr3.dip)
        tl = 0.5 * sr3.width * np.cos(sr3.dip)
        sr3.e_m = sr3.e + np.cos(ccw_to_x_up_dip) * tl
        sr3.n_m = sr3.n + np.sin(ccw_to_x_up_dip) * tl

        sr4 = deepcopy(sr2)
        sr4.y += sr2.width
        sr4.e = e + sr3.length * 1.5 * np.cos(self.ccw_to_x_stk) + np.cos(self.ccw_to_x_dip) * np.cos(
            sr3.dip) * sr3.width * 2
        sr4.n = n + sr3.length * 1.5 * np.sin(self.ccw_to_x_stk) + np.sin(self.ccw_to_x_dip) * np.cos(
            sr3.dip) * sr3.width * 2
        sr4.depth = z + np.sin(sr4.dip) * sr4.width *2

        sr4.depth_t = sr4.depth - sr4.width * np.sin(sr4.dip)
        tl = sr4.width * np.cos(sr4.dip)
        ccw_to_x_up_dip = sr4.ccw_to_x_dip + np.pi
        sr4.e_t = sr4.e + np.cos(ccw_to_x_up_dip) * tl
        sr4.n_t = sr4.n + np.sin(ccw_to_x_up_dip) * tl
        sr4.depth_m = sr4.depth - 0.5 * sr4.width * np.sin(sr4.dip)
        tl = 0.5 * sr4.width * np.cos(sr4.dip)
        sr4.e_m = sr4.e + np.cos(ccw_to_x_up_dip) * tl
        sr4.n_m = sr4.n + np.sin(ccw_to_x_up_dip) * tl
        return sr1, sr3, sr2, sr4


    def __str__(self):
        return '''strike = {}
dip = {}
length = {}
width = {}
ss = {}
ds = {}
ts = {}
open = {}
e = {}
n = {}
depth = {}
x = {}
y = {}'''.format(self.strike, self.dip, self.length, self.width, self.ss, self.ds, self.ts, self.open, self.e, self.n, self.depth, self.x, self.y)

    def __eq__(self, other):
        return self.strike == other.strike and self.dip == other.dip and self.length == other.length and self.e ==\
               other.e and self.n == other.n and self.depth == other.depth and self.x == other.x and self.y == other.y

    def adjacent(self, other):
        def eq(x, y):
            return np.abs(x - y) < 1e-10

        def biger(x, y):
            if eq(x, y):
                return False
            return x > y

        x1_left = self.x - 0.5 * self.length
        x1_right = x1_left + self.length
        x2_left = other.x - 0.5 * other.length
        x2_right = x2_left + other.length
        y1_boutom = self.y
        y1_up = y1_boutom - self.width
        y2_boutom = other.y
        y2_up = y2_boutom - other.width
        if self == other:
            return False
        elif biger(x1_left, x2_right) or biger(x2_left, x1_right):
            return False
        elif biger(y2_up, y1_boutom) or biger(y1_up, y2_boutom):
            return False
        elif (eq(x1_left, x2_right) and eq(y1_boutom, y2_up)) \
                or (eq(x1_left, x2_right) and eq(y1_up, y2_boutom)) \
                or (eq(x1_right, x2_left) and eq(y1_boutom, y2_up)) \
                or (eq(x1_right, x2_left) and eq(y1_up, y2_boutom)):
            return False
        else:
            return True
    def get_corners(self):
        return (self.p1, self.p2, self.p3, self.p4)

    def same_level(self, other):
        self_top = self.depth - self.width * np.sin(self.dip)
        other_top = other.depth - other.width * np.sin(other.dip)
        return not(self.depth <= other_top or self_top >= other.depth)

    def okada(self, x, y, z, strike_element, dip_element):

        def get_rot_mat(theta):
            rot = np.zeros((3, 3))
            rot[2, 2] = 1.0
            rot[0:2, 0:2] = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
            return rot
        theta = np.pi/2 - self.strike
        cords = get_rot_mat(theta).dot(np.array([x - self.e, y - self.n, z]))
        success, u, grad_u = dc3dwrapper(2.0 / 3.0, [cords[0], cords[1], z],
                                         self.depth, np.rad2deg(self.dip), [-self.length/2, self.length/2], [0, self.width],
                                         [self.strike_slip * strike_element, self.dip_slip * dip_element, 0.0])
        inv_rot = get_rot_mat(-theta)
        return inv_rot.dot(u), inv_rot.dot(grad_u)

    def stress(self, x, y, z, strike_element, dip_element, lambda_l, shaer_m):
        stress = np.zeros((3,3))
        okada_stress(self.e_t, self.n_t, self.depth_t, self.ccw_to_x_stk, self.dip, self.length, self.width,
                     self.strike_slip*strike_element, self.dip_slip * dip_element, 0, x, y, z, stress, lambda_l, shaer_m)
        return stress

    def stress_thread(self, x, y, z, strike_element, dip_element, lambda_l, shaer_m):
        return okada_stress_thread(self.e_t, self.n_t, self.depth_t, self.ccw_to_x_stk, self.dip, self.length, self.width,
                     self.strike_slip*strike_element, self.dip_slip * dip_element, 0, x, y, z, lambda_l, shaer_m, x.shape[0])
    def strain_thread(self, x, y, z, strike_element, dip_element, lambda_l, shaer_m):
        return okada_strain_thread(self.e_t, self.n_t, self.depth_t, self.ccw_to_x_stk, self.dip, self.length, self.width,
                     self.strike_slip*strike_element, self.dip_slip * dip_element, 0, x, y, z, lambda_l, shaer_m, x.shape[0])


    def to_gmt(self, slip):
        ccw_to_x_stk = np.pi / 2 - self.strike  # the angle betuen the fualt and the x axis cunter clock wise
        ccw_to_x_dip = -self.strike
        x1 = self.e + self.length / 2.0 * np.cos(ccw_to_x_stk)
        y1 = self.n + self.length / 2.0 * np.sin(ccw_to_x_stk)
        x2 = self.e - self.length / 2.0 * np.cos(ccw_to_x_stk)
        y2 = self.n - self.length / 2.0 * np.sin(ccw_to_x_stk)
        z1 = self.depth
        z2 = z1 - self.width * np.sin(self.dip)
        l = self.width * np.cos(self.dip)
        x3 = x1 - l * np.cos(ccw_to_x_dip)
        y3 = y1 - l * np.sin(ccw_to_x_dip)
        x4 = x2 - l * np.cos(ccw_to_x_dip)
        y4 = y2 - l * np.sin(ccw_to_x_dip)
        return \
'''> -Z%f
%f %f %f
%f %f %f
%f %f %f
%f %f %f
''' %(slip, x4, y4, z2, x3, y3, z2, x1, y1, z1, x2, y2, z1)

    def to_gmt_plain(self, slip, normal):
        ccw_to_x_stk = np.pi / 2 - self.strike  # the angle betuen the fualt and the x axis cunter clock wise
        ccw_to_x_dip = -self.strike
        x1 = self.e + self.length / 2.0 * np.cos(ccw_to_x_stk)
        y1 = self.n + self.length / 2.0 * np.sin(ccw_to_x_stk)
        x2 = self.e - self.length / 2.0 * np.cos(ccw_to_x_stk)
        y2 = self.n - self.length / 2.0 * np.sin(ccw_to_x_stk)
        z1 = self.depth
        z2 = z1 - self.width * np.sin(self.dip)
        l = self.width * np.cos(self.dip)
        x3 = x1 - l * np.cos(ccw_to_x_dip)
        y3 = y1 - l * np.sin(ccw_to_x_dip)
        x4 = x2 - l * np.cos(ccw_to_x_dip)
        y4 = y2 - l * np.sin(ccw_to_x_dip)
        a1 = np.array([x1, y1]).dot(normal)
        a2 = np.array([x2, y2]).dot(normal)
        a3 = np.array([x3, y3]).dot(normal)
        a4 = np.array([x4, y4]).dot(normal)
        return \
'''> -Z%f
%f %f
%f %f
%f %f
%f %f
''' % (slip, a4, -z2, a3, -z2, a1, -z1, a2, -z1)

    def normal(self):
        nz = np.cos(self.dip)
        nx = np.cos(self.strike) * np.sin(self.dip)
        ny = -np.sin(self.strike ) * np.sin(self.dip)
        return np.array([nx, ny, nz]).reshape(-1, 1)

    def shear_hat(self, rake):
        l_s = np.cos(rake)
        l_d = np.sin(rake)
        nz = -np.sin(self.dip) * l_d
        l = np.cos(self.dip) * l_d

        x_s = np.cos(self.strike) * l_s
        y_s = np.sin(self.strike) * l_s

        x_d = np.cos(self.strike + np.pi / 2) * l
        y_d = np.sin(self.strike + np.pi / 2) * l
        return np.array([x_s + x_d, y_s + y_d, nz]).reshape(-1, 1)



