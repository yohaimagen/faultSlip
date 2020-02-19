import cython

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

# declare the interface to the C code
cdef extern void pscokada(double x1, double y1, double z1, double strike1, double dip1, double L, double W, double slip_strike, double slip_dip, double open, double x2, double y2, double z2, double *sxx, double *syy, double *szz, double *sxy, double *syz, double *szx, double lame_lambda, double mu)
cdef extern void c_okada_stress(double x1, double y1, double z1, double strike1, double dip1, double L, double W, double slip_strike, double slip_dip, double open,
		double x2, double y2, double z2, double *s,
		double lame_lambda, double mu)
cdef extern void c_okada_stress_thread(double x1, double y1, double z1, double strike, double dip, double L, double W, double slip_strike, double slip_dip, double open,
		double *x2, double *y2, double *z2, double *s,
		double lame_lambda, double mu, int pop_num, int thread_num)


@cython.boundscheck(False)
@cython.wraparound(False)
def py_pscokada(double x1, double y1, double z1, double strike1, double dip1, double L, double W, double slip_strike, double slip_dip, double open,
		double x2, double y2, double z2,
		  np.ndarray[double, ndim=1, mode="c"] sxx not None,
		  np.ndarray[double, ndim=1, mode="c"] syy not None,
		  np.ndarray[double, ndim=1, mode="c"] szz not None,
		  np.ndarray[double, ndim=1, mode="c"] sxy not None,
		  np.ndarray[double, ndim=1, mode="c"] syz not None,
		  np.ndarray[double, ndim=1, mode="c"] szx not None,
		  double lame_lambda, double mu):

    pscokada(x1, y1, z1, strike1, dip1, L,  W, slip_strike, slip_dip, open, x2, y2, z2, &sxx[0], &syy[0], &szz[0], &sxy[0], &syz[0], &szx[0], lame_lambda, mu)

    return None

def okada_stress(double x1, double y1, double z1, double strike1, double dip1, double L, double W, double slip_strike, double slip_dip, double open,
		double x2, double y2, double z2,
		  np.ndarray[double, ndim=2, mode="c"] s not None,
		  double lame_lambda, double mu):

    c_okada_stress(x1, y1, z1, strike1, dip1, L,  W, slip_strike, slip_dip, open, x2, y2, z2, &s[0, 0], lame_lambda, mu)

    return None

def okada_stress_thread(double x1, double y1, double z1, double strike1, double dip1, double L, double W, double slip_strike, double slip_dip, double open,
		  np.ndarray[double, ndim=1, mode="c"] x2 not None,
		  np.ndarray[double, ndim=1, mode="c"] y2 not None,
		  np.ndarray[double, ndim=1, mode="c"] z2 not None,
		  double lame_lambda, double mu, int pop_num):

    cdef np.ndarray[double, ndim=1, mode='c'] out = np.zeros((pop_num * 3 * 3), dtype = np.float)
    c_okada_stress_thread(x1, y1, z1, strike1, dip1, L,  W, slip_strike, slip_dip, open, &x2[0], &y2[0], &z2[0], &out[0], lame_lambda, mu, pop_num, 12)


    return out.reshape(pop_num, 3, 3)