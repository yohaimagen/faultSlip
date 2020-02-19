import cython

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

# declare the interface to the C code
cdef extern void c_dist(double *p1, double *p2, double *p3, double *p4, double *out)
cdef extern void c_neighbors(double *p1, double *p2, double *p3, double *t4, double *t1, double *t2, double *t3, double *t4, double *out)

@cython.boundscheck(False)
@cython.wraparound(False)
def dist(np.ndarray[double, ndim=1, mode="c"] p1 not None,
           np.ndarray[double, ndim=1, mode="c"] p2 not None,
           np.ndarray[double, ndim=1, mode="c"] p3 not None,
           np.ndarray[double, ndim=1, mode="c"] p4 not None):
    cdef np.ndarray[double, ndim=1, mode='c'] out = np.empty((1,), dtype = np.float)
    c_dist(&p1[0], &p2[0], &p3[0], &p4[0], &out[0])
    return out[0]


def neighbors(np.ndarray[double, ndim=1, mode="c"] p1 not None,
              np.ndarray[double, ndim=1, mode="c"] p2 not None,
              np.ndarray[double, ndim=1, mode="c"] p3 not None,
              np.ndarray[double, ndim=1, mode="c"] p4 not None,
              np.ndarray[double, ndim=1, mode="c"] t1 not None,
              np.ndarray[double, ndim=1, mode="c"] t2 not None,
              np.ndarray[double, ndim=1, mode="c"] t3 not None,
              np.ndarray[double, ndim=1, mode="c"] t4 not None):
    cdef np.ndarray[double, ndim=1, mode='c'] out = np.empty((1,), dtype = np.float)
    c_neighbors(&p1[0], &p2[0], &p3[0], &p4[0], &t1[0], &t2[0], &t3[0], &t4[0], &out[0])
    return out[0] == 1