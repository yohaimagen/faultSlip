import cython


# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

# declare the interface to the C code


cdef extern void pscokada(double x1, double y1, double z1, double strike1, double dip1, double L, double W, double slip_strike, double slip_dip, double open,
		double x2, double y2, double z2, double *ux, double *uy, double *uz, double *sxx, double *syy, double *szz, double *sxy, double *syz, double *szx,
		double lame_lambda, double mu)





def psokada(double x1, double y1, double z1, double strike1, double dip1, double L, double W, double slip_strike,
            double slip_dip, double open, double x2, double y2, double z2,
            double lame_lambda, double mu):
    cdef np.ndarray[double, ndim=1, mode='c'] out_u = np.zeros(3, dtype=np.float)
    cdef np.ndarray[double, ndim=2, mode='c'] out_s = np.zeros((3, 3), dtype=np.float)
    pscokada(x1, y1, z1, strike1, dip1, L, W, slip_strike, slip_dip, open, x2, y2, z2,
             &out_u[0], &out_u[1], &out_u[2],
             &out_s[0, 0], &out_s[1, 1], &out_s[2, 2], &out_s[0, 1], &out_s[1, 2], &out_s[2, 0],
             lame_lambda, mu)
    out_s[1, 0] = out_s[0, 1]
    out_s[2, 1] = out_s[1, 2]
    out_s[0, 2] = out_s[2, 0]
    return out_u, out_s





