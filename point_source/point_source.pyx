import cython

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

# declare the interface to the C code
cdef extern void disp_point_source(float x1, float y1, float z1, float strike1, float dip1, float moment_strike, float moment_dip, float moment_inflation,
		float moment_open, float x2, float y2, float z2, float *sx, float *sy, float *sz,
		float lame_lambda, float mu)
cdef extern void c_point_source_strain(float x1, float y1, float z1, float strike1, float dip1, float moment_strike, float moment_dip, float moment_inflation,
		float moment_open, float x2, float y2, float z2, float *s,
		float lame_lambda, float mu);
cdef extern void c_point_source_stress(float x1, float y1, float z1, float strike1, float dip1, float moment_strike, float moment_dip, float moment_inflation,
		float moment_open, float x2, float y2, float z2, float *s,
		float lame_lambda, float mu)
cdef extern void c_ps_stress(float x1, float y1, float z1, float strike1, float dip1, float moment_strike, float moment_dip, float moment_inflation,
		float moment_open, float *x2, float *y2, float *z2, float *s, float lame_lambda, float mu, int pop_num, int thread_num)
cdef extern void c_ps_strain(float x1, float y1, float z1, float strike1, float dip1, float moment_strike, float moment_dip, float moment_inflation,
		float moment_open, float *x2, float *y2, float *z2, float *s, float lame_lambda, float mu, int pop_num, int thread_num)


@cython.boundscheck(False)
@cython.wraparound(False)
def ps_strain(float x1, float y1, float z1, float strike1, float dip1, float moment_strike, float moment_dip, float moment_inflation, float moment_open,
          np.ndarray[float, ndim=1, mode="c"] x2 not None,
		  np.ndarray[float, ndim=1, mode="c"] y2 not None,
		  np.ndarray[float, ndim=1, mode="c"] z2 not None,
          float lame_lambda, float mu):
    pop_num = x2.shape[0]
    cdef np.ndarray[float, ndim=1, mode='c'] out = np.zeros(9 * pop_num, dtype = np.single)
    c_ps_strain(x1, y1, z1, strike1, dip1, moment_strike, moment_dip, moment_inflation, moment_open, &x2[0], &y2[0], &z2[0], &out[0], lame_lambda, mu, pop_num, 12)
    return out.reshape(pop_num, 3, 3)

@cython.boundscheck(False)
@cython.wraparound(False)
def ps_stress(float x1, float y1, float z1, float strike1, float dip1, float moment_strike, float moment_dip, float moment_inflation, float moment_open,
          np.ndarray[float, ndim=1, mode="c"] x2 not None,
		  np.ndarray[float, ndim=1, mode="c"] y2 not None,
		  np.ndarray[float, ndim=1, mode="c"] z2 not None,
          float lame_lambda, float mu):
    pop_num = x2.shape[0]
    cdef np.ndarray[float, ndim=1, mode='c'] out = np.zeros(9 * pop_num, dtype = np.single)
    c_ps_stress(x1, y1, z1, strike1, dip1, moment_strike, moment_dip, moment_inflation, moment_open, &x2[0], &y2[0], &z2[0], &out[0], lame_lambda, mu, pop_num, 12)
    return out.reshape(pop_num, 3, 3)


@cython.boundscheck(False)
@cython.wraparound(False)
def py_disp_point_source(float x1, float y1, float z1, float strike1, float dip1, float moment_strike, float moment_dip, float moment_inflation,
		 float moment_open, float x2, float y2, float z2, float lame_lambda, float mu):

    cdef np.ndarray[float, ndim=1, mode='c'] out = np.zeros(3, dtype = np.single)
    disp_point_source(x1, y1, z1, strike1, dip1, moment_strike, moment_dip, moment_inflation, moment_open, x2, y2, z2, &out[0], &out[1], &out[2], lame_lambda, mu)

    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def py_point_source_strain(float x1, float y1, float z1, float strike1, float dip1, float moment_strike, float moment_dip, float moment_inflation,
		float moment_open, float x2, float y2, float z2, float lame_lambda, float mu):

    cdef np.ndarray[float, ndim=1, mode='c'] out = np.zeros(9, dtype = np.single)
    c_point_source_strain(x1, y1, z1, strike1, dip1, moment_strike, moment_dip, moment_inflation, moment_open, x2, y2, z2, &out[0], lame_lambda, mu)
    return out.reshape(3, 3)

@cython.boundscheck(False)
@cython.wraparound(False)
def py_point_source_stress(float x1, float y1, float z1, float strike1, float dip1, float moment_strike, float moment_dip, float moment_inflation,
		float moment_open, float x2, float y2, float z2, float lame_lambda, float mu):

    cdef np.ndarray[float, ndim=1, mode='c'] out = np.zeros(9, dtype = np.single)
    c_point_source_stress(x1, y1, z1, strike1, dip1, moment_strike, moment_dip, moment_inflation, moment_open, x2, y2, z2, &out[0], lame_lambda, mu)
    return out.reshape(3, 3)


