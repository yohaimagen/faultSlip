"""
okada85.pyx

TODO add doc

"""

import cython

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

# declare the interface to the C code
cdef extern void c_disloc(double *pEOutput, double *pNOutput, double *pZOutput, double *pModel, double *pECoords, double *pNCoords, double nu, int NumStat, int NumDisl)
cdef extern void c_disloc_1d(double *pEOutput, double *pNOutput, double *pZOutput, double *pModel, double *pECoords, double *pNCoords, double nu, int NumStat, int NumDisl)
cdef extern void c_disloc_m(double *pEOutput, double *pNOutput, double *pZOutput, double *pModel, double *pECoords, double *pNCoords, double nu, int NumStat, int NumDisl, int thread_num)

@cython.boundscheck(False)
@cython.wraparound(False)
def disloc_2d(np.ndarray[double, ndim=2, mode="c"] pEOutput not None,
           np.ndarray[double, ndim=2, mode="c"] pNOutput not None,
           np.ndarray[double, ndim=2, mode="c"] pZOutput not None,
           np.ndarray[double, ndim=1, mode="c"] pModel not None,
           np.ndarray[double, ndim=2, mode="c"] pECoords not None,
           np.ndarray[double, ndim=2, mode="c"] pNCoords not None,
           double nu,int NumStat, int NumDisl ):
    """

    compute okada dislocation for 2_d array
  :param pEOutput: output array for the east west displacement
  :param pNOutput: output array for the north south displacement
  :param pZOutput: output array for the east west displacement
  :param pModel: the source model in the order pModel[1...] = length, width, depth, dip(deg), strike(deg), easting, northing, strike-slip, dip-slip, tensile
  :param pECoords: east cordination of the stations
  :param pNCoords: north cordination of the stations
  :param nu: poisson ratio
  :param NumStat: number of station
  :param NumDisl: number of sources
  :return: the displacement in coordination pEcoords,pNcoords due to okada finte rectanle suorce as describe in pModel
    """
    c_disloc_m(&pEOutput[0,0], &pNOutput[0,0], &pZOutput[0,0], &pModel[0], &pECoords[0,0], &pNCoords[0,0], nu, NumStat, NumDisl, 12)

    return None

def disloc_1d(np.ndarray[double, ndim=1, mode="c"] pEOutput not None,
           np.ndarray[double, ndim=1, mode="c"] pNOutput not None,
           np.ndarray[double, ndim=1, mode="c"] pZOutput not None,
           np.ndarray[double, ndim=1, mode="c"] pModel not None,
           np.ndarray[double, ndim=1, mode="c"] pECoords not None,
           np.ndarray[double, ndim=1, mode="c"] pNCoords not None,
           double nu,int NumStat, int NumDisl ):
  """
  compute okada dislocation for 1_d array
  :param pEOutput: output array for the east west displacement
  :param pNOutput: output array for the north south displacement
  :param pZOutput: output array for the east west displacement
  :param pModel: the source model in the order pModel[1...] = length, width, depth, dip(deg), strike(deg), easting, northing, strike-slip, dip-slip, tensile
  :param pECoords: east cordination of the stations
  :param pNCoords: north cordination of the stations
  :param nu: poisson ratio
  :param NumStat: number of station
  :param NumDisl: number of sources
  :return: the displacement in coordination pEcoords,pNcoords due to okada finte rectanle suorce as describe in pModel
  """
  c_disloc_m(&pEOutput[0], &pNOutput[0], &pZOutput[0], &pModel[0], &pECoords[0], &pNCoords[0], nu, NumStat, NumDisl, 12)
  return None

def disloc_1ds(np.ndarray[double, ndim=1, mode="c"] pEOutput not None,
           np.ndarray[double, ndim=1, mode="c"] pNOutput not None,
           np.ndarray[double, ndim=1, mode="c"] pZOutput not None,
           np.ndarray[double, ndim=1, mode="c"] pModel not None,
           np.ndarray[double, ndim=1, mode="c"] pECoords not None,
           np.ndarray[double, ndim=1, mode="c"] pNCoords not None,
           double nu,int NumStat, int NumDisl ):
  """
  compute okada dislocation for 1_d array
  :param pEOutput: output array for the east west displacement
  :param pNOutput: output array for the north south displacement
  :param pZOutput: output array for the east west displacement
  :param pModel: the source model in the order pModel[1...] = length, width, depth, dip(deg), strike(deg), easting, northing, strike-slip, dip-slip, tensile
  :param pECoords: east cordination of the stations
  :param pNCoords: north cordination of the stations
  :param nu: poisson ratio
  :param NumStat: number of station
  :param NumDisl: number of sources
  :return: the displacement in coordination pEcoords,pNcoords due to okada finte rectanle suorce as describe in pModel
  """
  c_disloc_1d(&pEOutput[0], &pNOutput[0], &pZOutput[0], &pModel[0], &pECoords[0], &pNCoords[0], nu, NumStat, NumDisl)
  return None


def disloc_1d_m(np.ndarray[double, ndim=1, mode="c"] pEOutput not None,
           np.ndarray[double, ndim=1, mode="c"] pNOutput not None,
           np.ndarray[double, ndim=1, mode="c"] pZOutput not None,
           np.ndarray[double, ndim=1, mode="c"] pModel not None,
           np.ndarray[double, ndim=1, mode="c"] pECoords not None,
           np.ndarray[double, ndim=1, mode="c"] pNCoords not None,
           double nu,int NumStat, int NumDisl ):
  """
  compute okada dislocation for 1_d array
  :param pEOutput: output array for the east west displacement
  :param pNOutput: output array for the north south displacement
  :param pZOutput: output array for the east west displacement
  :param pModel: the source model in the order pModel[1...] = length, width, depth, dip(deg), strike(deg), easting, northing, strike-slip, dip-slip, tensile
  :param pECoords: east cordination of the stations
  :param pNCoords: north cordination of the stations
  :param nu: poisson ratio
  :param NumStat: number of station
  :param NumDisl: number of sources
  :return: the displacement in coordination pEcoords,pNcoords due to okada finte rectanle suorce as describe in pModel
  """
  c_disloc_m(&pEOutput[0], &pNOutput[0], &pZOutput[0], &pModel[0], &pECoords[0], &pNCoords[0], nu, NumStat, NumDisl, 12)
  return None

@cython.boundscheck(False)
@cython.wraparound(False)
def disloc_2d_m(np.ndarray[double, ndim=2, mode="c"] pEOutput not None,
           np.ndarray[double, ndim=2, mode="c"] pNOutput not None,
           np.ndarray[double, ndim=2, mode="c"] pZOutput not None,
           np.ndarray[double, ndim=1, mode="c"] pModel not None,
           np.ndarray[double, ndim=2, mode="c"] pECoords not None,
           np.ndarray[double, ndim=2, mode="c"] pNCoords not None,
           double nu,int NumStat, int NumDisl ):
    """
    TODO add doc

    """
    c_disloc_m(&pEOutput[0,0], &pNOutput[0,0], &pZOutput[0,0], &pModel[0], &pECoords[0,0], &pNCoords[0,0], nu, NumStat, NumDisl, 12)

    return None