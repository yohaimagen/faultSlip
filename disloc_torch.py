import numpy as np
import torch
from disloc import disloc

def disloc_pytorch(length, width, depth, dip, strike, easting, northing, ss, ds, ts, e, n, nu):
    _cos_dip = torch.cos(dip)
    _sin_dip = torch.sin(dip)
    if (length < 0 or width < 0 or depth < 0 or (depth - _sin_dip * width) < 0):
        # return torch.zeros_like(e),\
        #        torch.zeros_like(e),\
        #        torch.zeros_like(e)
        return torch.ones_like(e) * length * width * depth * dip * strike * easting * northing  * ts * 1e20, \
               torch.ones_like(e) * length * width * depth * dip * strike * easting * northing  * ts * 1e20, \
               torch.ones_like(e) * length * width * depth * dip * strike * easting * northing  * ts * 1e20
        # raise Exception('the dislocation is unphysical')
    if torch.abs(_cos_dip) < 2.2204460492503131e-16:
        cos_dip = 0.0
        if _sin_dip > 0:
            sin_dip = 1.0
        else:
            sin_dip = 0.0
    else:
        cos_dip = _cos_dip
        sin_dip = _sin_dip
    angle = -(np.pi / 2 - strike)
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)

    relt_e = cos_angle * (e - easting) - sin_angle * (n - northing) + 0.5 * length
    relt_n = sin_angle * (e - easting) + cos_angle * (n - northing)
    SS_0, SS_1, SS_2, DS_0, DS_1, DS_2, TS_0, TS_1, TS_2 = okada_torch(1 - 2 * nu, sin_dip, cos_dip, length, width, depth,
                       relt_e, relt_n, ss, ds, ts)

    x = SS_0 + DS_0 + TS_0
    y = SS_1 + DS_1 + TS_1
    return cos_angle * x + sin_angle * y, -sin_angle * x + cos_angle * y, SS_2 + DS_2 + TS_2


def okada_torch(alp, sin_dip, cos_dip, length, width, depth, X, Y, ss, ds, ts):
    PI2INV = 0.15915494309189535


    depth_sin_dip = depth * sin_dip
    depth_cos_dip = depth * cos_dip

    p = Y * cos_dip + depth_sin_dip
    q = Y * sin_dip - depth_cos_dip
    SS_0, SS_1, SS_2, DS_0, DS_1, DS_2, TS_0, TS_1, TS_2 = a1(p - width, PI2INV, X, q, sin_dip, cos_dip, alp, ss, ds, ts, length)
    SS_0_t, SS_1_t, SS_2_t, DS_0_t, DS_1_t, DS_2_t, TS_0_t, TS_1_t, TS_2_t = a1(p, -PI2INV, X, q, sin_dip, cos_dip, alp, ss, ds, ts, length)


    return SS_0 + SS_0_t, SS_1 + SS_1_t, SS_2 + SS_2_t, DS_0 + DS_0_t, DS_1 + DS_1_t, DS_2 + DS_2_t, TS_0 + TS_0_t, TS_1 + TS_1_t, TS_2 + TS_2_t


def a1(et, sign, X, q, sin_dip, cos_dip, alp, ss, ds, ts, length):
    SS_0, SS_1, SS_2, DS_0, DS_1, DS_2, TS_0, TS_1, TS_2 = a2(X - length, sign, et, q, sin_dip, cos_dip, alp, ss, ds, ts)
    SS_0, SS_1, SS_2, DS_0, DS_1, DS_2 = -SS_0, -SS_1, -SS_2, -DS_0, -DS_1, -DS_2
    SS_0_t, SS_1_t, SS_2_t, DS_0_t, DS_1_t, DS_2_t, TS_0_t, TS_1_t, TS_2_t = a2(X, -sign, et, q, sin_dip, cos_dip, alp, ss, ds, ts)
    return SS_0 - SS_0_t, SS_1 - SS_1_t, SS_2 - SS_2_t, DS_0 -DS_0_t, DS_1 - DS_1_t, DS_2 - DS_2_t, TS_0 + TS_0_t, TS_1 + TS_1_t, TS_2 + TS_2_t

def a2(xi, sign, et, q, sin_dip, cos_dip, alp, ss, ds, ts):
    sin_cos_dip = sin_dip * cos_dip
    sin_2_dip = sin_dip * sin_dip
    xi2 = xi.pow(2)
    et2 = et.pow(2)
    q2 = q.pow(2)
    r2 = xi2 + et2 + q2
    r = torch.sqrt(r2)
    d = et * sin_dip - q * cos_dip
    y = et * cos_dip + q * sin_dip
    ret = r + et
    mask = ret < 0.0
    ret[mask] = 0.0
    rd = r + d
    tt = torch.atan(xi * et / (q * r))
    tt[q == 0.0] = 0.0
    re = 1 / ret
    dle = torch.log(ret)
    re[ret == 0.0] = 0.0
    dle[ret == 0.0] = -torch.log(r[ret == 0.0] - et[ret == 0.0])
    rrx = 1 / (r * (r + xi))
    rre = re / r
    if cos_dip == 0.0:
        rd2 = rd * rd
        a1 = -alp / 2 * xi * q / rd2
        a3 = alp / 2 * (et / rd + y * q / rd2 - dle)
        a4 = -alp * q / rd
        a5 = -alp * xi * sin_dip / rd
    else:
        td = sin_dip / cos_dip
        x = torch.sqrt(xi2 + q2)
        a5 = alp * 2 / cos_dip * torch.atan(
            (et * (x + q * cos_dip) + x * (r + x) * sin_dip) / (xi * (r + x) * cos_dip))
        a5[xi == 0.0] = 0.0
        a4 = alp / cos_dip * (torch.log(rd) - sin_dip * dle)
        a3 = alp * (y / rd / cos_dip - dle) + td * a4
        a1 = -alp / cos_dip * xi / rd - td * a5
    a2 = -alp * dle - a3
    req = rre * q
    rxq = rrx * q

    mult = sign * ss
    SS_0 =  mult * (req * xi + tt + a1 * sin_dip)
    SS_1 =  mult * (req * y + q * cos_dip * re + a2 * sin_dip)
    SS_2 =  mult * (req * d + q * sin_dip * re + a4 * sin_dip)

    mult = sign * ds
    DS_0 = mult * (q / r - a3 * sin_cos_dip)
    DS_1 = mult * (y * rxq + cos_dip * tt - a1 * sin_cos_dip)
    DS_2 = mult * (d * rxq + sin_dip * tt - a5 * sin_cos_dip)

    mult = sign * ts
    TS_0 = mult * (q2 * rre - a3 * sin_2_dip)
    TS_1 = mult * (-d * rxq - sin_dip * (xi * q * rre - tt) - a1 * sin_2_dip)
    TS_2 = mult * (y * rxq + cos_dip * (xi * q * rre - tt) - a5 * sin_2_dip)
    return SS_0, SS_1, SS_2, DS_0, DS_1, DS_2, TS_0, TS_1, TS_2


def disloc_py(length, width, depth, dip, strike, easting, northing, ss, ds, ts, e, n, nu):
    cos_dip = np.cos(dip)
    sin_dip = np.sin(dip)
    if (length < 0 or width < 0 or depth < 0 or (depth - sin_dip * width) < 1e-12):
        return 0, 0, 0
    if np.abs(cos_dip) < 2.2204460492503131e-16:
        cos_dip = 0.0
        if sin_dip > 0:
            sin_dip = 1.0
        else:
            sin_dip = 0.0
    angle = -(np.pi / 2 - strike)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    SS = np.zeros(3)
    DS = np.zeros(3)
    TS = np.zeros(3)
    relt_e = cos_angle * (e - easting) - sin_angle * (n - northing) + 0.5 * length
    relt_n = sin_angle * (e - easting) + cos_angle * (n - northing)
    SS, DS, TS = okada(1 - 2 * nu, sin_dip, cos_dip, length, width, depth,
                       relt_e, relt_n, ss, ds, ts)

    x = SS[0] + DS[0] + TS[0]
    y = SS[1] + DS[1] + TS[1]
    return cos_angle * x + sin_angle * y, -sin_angle * x + cos_angle * y, SS[2] + DS[2] + TS[2]


def okada(alp, sin_dip, cos_dip, length, width, depth, X, Y, ss, ds, ts):
    ala = np.zeros(2)
    awa = np.zeros(2)
    SS = np.zeros(3)
    DS = np.zeros(3)
    TS = np.zeros(3)
    PI2INV = 0.15915494309189535

    ala[0] = length

    awa[0] = width

    sin_cos_dip = sin_dip * cos_dip
    sin_2_dip = sin_dip * sin_dip
    depth_sin_dip = depth * sin_dip
    depth_cos_dip = depth * cos_dip

    p = Y * cos_dip + depth_sin_dip
    q = Y * sin_dip - depth_cos_dip


    for k in range(2):
        et = p - awa[k]
        for j in range(2):
            sign = PI2INV
            xi = X - ala[j]
            if (j + k == 1):
                sign = -PI2INV
            xi2 = xi ** 2
            et2 = et ** 2
            q2 = q ** 2
            r2 = xi2 + et2 + q2
            r = np.sqrt(r2)
            r3 = r * r2
            d = et * sin_dip - q * cos_dip
            y = et * cos_dip + q * sin_dip
            ret = r + et
            if ret < 0.0:
                ret = 0.0
            rd = r + d
            if q != 0.0:
                tt = np.arctan(xi * et / (q * r))
            else:
                tt = 0.0
            if ret != 0.0:
                re = 1 / ret
                dle = np.log(ret)
            else:
                re = 0.0
                dle = -np.log(r - et)
            rrx = 1 / (r * (r + xi))
            rre = re / r
            if cos_dip == 0.0:
                rd2 = rd * rd
                a1 = -alp / 2 * xi * q / rd2
                a3 = alp / 2 * (et / rd + y * q / rd2 - dle)
                a4 = -alp * q / rd
                a5 = -alp * xi * sin_dip / rd
            else:
                td = sin_dip / cos_dip
                x = np.sqrt(xi2 + q2)
                if xi == 0.0:
                    a5 = 0.0
                else:
                    a5 = alp * 2 / cos_dip * np.arctan(
                        (et * (x + q * cos_dip) + x * (r + x) * sin_dip) / (xi * (r + x) * cos_dip))
                a4 = alp / cos_dip * (np.log(rd) - sin_dip * dle)
                a3 = alp * (y / rd / cos_dip - dle) + td * a4
                a1 = -alp / cos_dip * xi / rd - td * a5
            a2 = -alp * dle - a3
            req = rre * q
            rxq = rrx * q
            if ss != 0.0:
                mult = sign * ss
                SS[0] -= mult * (req * xi + tt + a1 * sin_dip)
                SS[1] -= mult * (req * y + q * cos_dip * re + a2 * sin_dip)
                SS[2] -= mult * (req * d + q * sin_dip * re + a4 * sin_dip)
            if ds != 0.0:
                mult = sign * ds
                DS[0] -= mult * (q / r - a3 * sin_cos_dip)
                DS[1] -= mult * (y * rxq + cos_dip * tt - a1 * sin_cos_dip)
                DS[2] -= mult * (d * rxq + sin_dip * tt - a5 * sin_cos_dip)
            if ts != 0.0:
                mult = sign * ts
                TS[0] += mult * (q2 * rre - a3 * sin_2_dip)
                TS[1] += mult * (-d * rxq - sin_dip * (xi * q * rre - tt) - a1 * sin_2_dip)
                TS[2] += mult * (y * rxq + cos_dip * (xi * q * rre - tt) - a5 * sin_2_dip)
    return SS, DS, TS

def disloc_numpy(length, width, depth, dip, strike, easting, northing, ss, ds, ts, e, n, nu):
    cos_dip = np.cos(dip)
    sin_dip = np.sin(dip)
    un_phisical = np.concatenate((np.argwhere(length < 0), np.argwhere(width < 0), np.argwhere(depth < 0), np.argwhere((depth - sin_dip * width) < 1e-12)))
    if un_phisical.shape[0] > 0:
        raise Exception('the following dislocations are unphysical' + str(np.unique(un_phisical)))


    mask = np.argwhere(np.abs(cos_dip) < 2.2204460492503131e-16)
    cos_dip[mask] = 0.0
    sin_dip[np.intersect1d(mask, np.argwhere(sin_dip > 0))] = 1.0
    sin_dip[np.intersect1d(mask, np.argwhere(sin_dip <= 0))] = 0.0

    angle = -(np.pi / 2 - strike).reshape(-1, 1)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)


    relt_e = cos_angle.reshape(-1, 1) * (-np.tile(easting.reshape(-1, 1), (1, e.shape[0])) + e) - sin_angle.reshape(-1, 1) * (-np.tile(northing.reshape(-1, 1), (1, n.shape[0])) + n) + 0.5 * length.reshape(-1, 1)
    relt_n = sin_angle.reshape(-1, 1) * (-np.tile(easting.reshape(-1, 1), (1, e.shape[0])) + e) + cos_angle.reshape(-1, 1) * (
                -np.tile(northing.reshape(-1, 1), (1, n.shape[0])) + n)
    SS, DS, TS = okada_nunpy(1 - 2 * nu, sin_dip, cos_dip, length, width, depth,
                       relt_e, relt_n, ss, ds, ts)

    x = SS[0] + DS[0] + TS[0]
    y = SS[1] + DS[1] + TS[1]
    return np.array([cos_angle * x + sin_angle * y, -sin_angle * x + cos_angle * y, SS[2] + DS[2] + TS[2]])


def okada_nunpy(alp, sin_dip, cos_dip, length, width, depth, X, Y, ss, ds, ts):
    ala = []
    awa = []
    SS = np.zeros((3, X.shape[0], X.shape[1]))
    DS = np.zeros((3, X.shape[0], X.shape[1]))
    TS = np.zeros((3, X.shape[0], X.shape[1]))
    PI2INV = 0.15915494309189535



    ala.append(length)
    ala.append(np.zeros(length.shape))

    awa.append(width)
    awa.append(np.zeros(width.shape))

    sin_cos_dip = sin_dip * cos_dip
    sin_2_dip = sin_dip * sin_dip
    depth_sin_dip = depth * sin_dip
    depth_cos_dip = depth * cos_dip

    p = Y * cos_dip.reshape(-1, 1) + depth_sin_dip.reshape(-1, 1)
    q = Y * sin_dip.reshape(-1, 1) - depth_cos_dip.reshape(-1, 1)


    for k in range(2):
        et = p - awa[k].reshape(-1, 1)
        for j in range(2):
            sign = PI2INV
            xi = X - ala[j].reshape(-1, 1)
            if (j + k == 1):
                sign = -PI2INV
            xi2 = xi ** 2
            et2 = et ** 2
            q2 = q ** 2
            r2 = xi2 + et2 + q2
            r = np.sqrt(r2)
            d = et * sin_dip.reshape(-1, 1) - q * cos_dip.reshape(-1, 1)
            y = et * cos_dip.reshape(-1, 1) + q * sin_dip.reshape(-1, 1)
            ret = r + et
            ret[ret < 0] = 0.0
            rd = r + d

            tt = np.arctan(xi * et / (q * r))
            tt[q == 0] = 0.0

            re = 1 / ret
            re[ret == 0.0] = 0.0
            dle = np.log(ret)
            dle[ret == 0.0] = -np.log(r[ret == 0.0] - et[ret == 0.0])


            rrx = 1 / (r * (r + xi))
            rre = re / r


            a1 = np.zeros(xi.shape)
            a2 = np.zeros(xi.shape)
            a3 = np.zeros(xi.shape)
            a4 = np.zeros(xi.shape)
            a5= np.zeros(xi.shape)

            rd2 = rd * rd

            mask = cos_dip == 0.0

            a1[mask] = -alp / 2 * xi[mask] * q[mask] / rd2[mask]
            a3[mask] = alp / 2 * (et[mask] / rd[mask] + y[mask] * q[mask] / rd2[mask] - dle[mask])
            a4[mask] = -alp * q[mask] / rd[mask]
            a5[mask] = -alp * xi[mask] * sin_dip[mask].reshape(-1, 1) / rd[mask]

            x = np.sqrt(xi2 + q2)
            mask = np.logical_not(mask)
            td = sin_dip[mask] / cos_dip[mask]

            a5[mask] = alp * 2 / cos_dip[mask].reshape(-1, 1) * np.arctan(
                (et[mask] * (x[mask] + q[mask] * cos_dip[mask].reshape(-1, 1)) +
                 x[mask] * (r[mask] + x[mask]) * sin_dip[mask].reshape(-1, 1)) /
                (xi[mask] * (r[mask] + x[mask]) * cos_dip[mask].reshape(-1, 1)))
            a5[mask][xi[mask] == 0.0] = 0.0

            a4[mask] = alp / cos_dip[mask].reshape(-1, 1) * (np.log(rd[mask]) - sin_dip[mask].reshape(-1, 1) * dle[mask])
            a3[mask] = alp * (y[mask] / rd[mask] / cos_dip[mask].reshape(-1, 1) - dle[mask]) + td.reshape(-1, 1) * a4[mask]
            a1[mask] = -alp / cos_dip[mask].reshape(-1, 1) * xi[mask] / rd[mask] - td.reshape(-1, 1) * a5[mask]




            a2 = -alp * dle - a3
            req = rre * q
            rxq = rrx * q

            mult = sign * ss.reshape(-1, 1)
            SS[0] -= mult * (req * xi + tt + a1 * sin_dip.reshape(-1, 1))
            SS[1] -= mult * (req * y + q * cos_dip.reshape(-1, 1) * re + a2 * sin_dip.reshape(-1, 1))
            SS[2] -= mult * (req * d + q * sin_dip.reshape(-1, 1) * re + a4 * sin_dip.reshape(-1, 1))

            mult = sign * ds.reshape(-1, 1)
            DS[0] -= mult * (q / r - a3 * sin_cos_dip.reshape(-1, 1))
            DS[1] -= mult * (y * rxq + cos_dip.reshape(-1, 1) * tt - a1 * sin_cos_dip.reshape(-1, 1))
            DS[2] -= mult * (d * rxq + sin_dip.reshape(-1, 1) * tt - a5 * sin_cos_dip.reshape(-1, 1))

            mult = sign * ts.reshape(-1, 1)
            TS[0] += mult * (q2 * rre - a3 * sin_2_dip.reshape(-1, 1))
            TS[1] += mult * (-d * rxq - sin_dip.reshape(-1, 1) * (xi * q * rre - tt) - a1 * sin_2_dip.reshape(-1, 1))
            TS[2] += mult * (y * rxq + cos_dip.reshape(-1, 1) * (xi * q * rre - tt) - a5 * sin_2_dip.reshape(-1, 1))
    return SS, DS, TS


# E = np.zeros(2)
# N = np.zeros(2)
# Z = np.zeros(2)
# model = np.array([12.35, 20.0, 14.14, np.rad2deg(0.5), np.rad2deg(2.0), 12.27, 15.8, 0.0, 0.0, 1.0])
# east = torch.DoubleTensor([10.0, 20.0])
# north = torch.DoubleTensor([10.0, 20.0])
# disloc.disloc_1d(E, N, Z,  model, east.numpy(), north.numpy(), 0.25, 2, 1)
# model[3:5] = np.deg2rad(model[3:5])
# model = torch.autograd.Variable(torch.from_numpy(model), requires_grad=True)
# disp = disloc_pytorch(model[0], model[1], model[2], model[3], model[4], model[5], model[6], model[7], model[8], model[9], east, north, 0.25)
#
# # print('E:%f, N:%f, Z:%f' %(E[0] - disp[0].detach().numpy(), N[0] - disp[1].detach().numpy(), Z[0]- disp[2].detach().numpy()))
# print(torch.cat(disp).shape)
# print(E, N, Z)
# print(disp)
#
#
# loss = torch.sum(disp[0]) + torch.sum(disp[1]) + torch.sum(disp[2])
# loss.backward()
# print(model.grad)
#
# length = np.array([10.0])
# width = np.array([10.0])
# depth = np.array([11.0])
# dip = np.deg2rad(np.array([90]))
# strike = np.deg2rad(np.array([30]))
# easting = np.array([20.0])
# northing = np.array([20.0])
# ss = np.array([1.0])
# ds = np.array([1.0])
# ts = np.array([1.0])
#
#
# length = np.array([10.0, 1.0, 2.0])
# width = np.array([10.0, 1.0, 10.0])
# depth = np.array([10.0, 10.0, 15.0])
# dip = np.deg2rad(np.array([45, 45, 90]))
# strike = np.deg2rad(np.array([30, 30, 30]))
# easting = np.array([20.0, 20.0, 20.0])
# northing = np.array([20.0, 30.0, 40.0])
# ss = np.array([1.0, 1.0, 1.0])
# ds = np.array([1.0, 1.0, 1.0])
# ts = np.array([1.0, 1.0, 1.0])
#
# e = np.array([10.0])
# n = np.array([10.0])
#
# e = np.array([10.0, 5.0])
# n = np.array([10.0, 5.0])
#
# nu = 0.25
# f = np.zeros((3, length.shape[0], e.shape[0]))
# for l, w, d, di, s, ea, no, _ss, _ds, _ts, i in zip(length, width, depth, dip, strike, easting, northing, ss, ds, ts, range(length.shape[0])):
#     for _e, _n, j in zip(e, n, range(e.shape[0])):
#         f[:,i, j] = disloc_py(l, w, d, di, s, ea, no, _ss, _ds, _ts, _e, _n, 0.25)
#
# k = disloc_numpy(length, width, depth, dip, strike, easting, northing, ss, ds, ts, e, n, nu)
#
# print(f - k)


