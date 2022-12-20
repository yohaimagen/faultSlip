import numpy as np
import xarray as xr

def m2dd(m, lat=0):
    """
    transform distance in meter to degrees

    Args:
        m: distance in meters
        lat: for converting longitude distances insert the latitude, default 0

    Returns:
        distance in degrees

    """
    return m / (111319.9 * np.cos(np.deg2rad(lat)))

def dd2m(dd, lat=0):
    """
    transform distance in degrees to meters

    Args:
        dd(float): distance in degrees
        lat(float): for converting longitude distances insert the latitude, default 0
    Returns:
        dist: distance in meters
    Examples
    --------

    >>> import numpy as np
    >>> degs = np.array([1.0, 2.0, 3.0])
    >>> dd2m(degs)
    array([111319.9, 222639.8, 333959.7])
    >>> dd2m(degs, lat=45)
    array([ 78715.05617101, 157430.11234202, 236145.16851303])
    """
    return dd * (111319.9 * np.cos(np.deg2rad(lat)))


def normal(strike, dip):
    """
        computer the nornal to the plane defined by `strike` and `dip`, in cartisean cordinatation


        Args:
            strike(float): strike in degrees clockwise from North
            dip(float): dip in degrees from the horizontal plain
        Returns:
            normal: numpy 1d array of shape (3, 1) normal to the plain defined by strike and dip
        Examples
    --------

    >>> import numpy as np
    >>> normal(0, 90)
    array([[ 1.000000e+00],
          [-0.000000e+00],
          [ 6.123234e-17]])
    >>> normal(270, 90)
    array([[-1.8369702e-16],
          [ 1.0000000e+00],
          [ 6.1232340e-17]])
    >>> normal(45, 45)
    array([[ 0.5       ],
          [-0.5       ],
          [ 0.70710678]])
    """
    strike = np.radians((strike))
    dip = np.radians((dip))
    nz = np.cos(dip)
    nx = np.cos(strike) * np.sin(dip)
    ny = -np.sin(strike) * np.sin(dip)
    return np.array([nx, ny, nz]).reshape(-1, 1)


def shear(strike, dip, rake):
    """
            computer the shear unit vector on the plane defined by `strike` and `dip` and in rake direction
             in cartisean cordinatation


            Args:
                strike(float): strike in degrees clockwise from North
                dip(float): dip in degrees from the horizontal plain
                rake(float): the rake angle, 0 been strike direction and 90 normal slip
            Returns:
                shear: numpy 1d array of shape (3, 1) paralel to the plain defined by strike and dip
                 and in the direction of rake
            Examples
        --------

        >>> import numpy as np
        >>> shear(0, 90, 0)
        array([[6.123234e-17],
              [1.000000e+00],
              [0.000000e+00]])
        >>> normal(0, 90, -90)
        array([[-6.123234e-17],
               [ 6.123234e-17],
               [ 1.000000e+00]])
        >>> normal(45, 45, 45)
        array([[ 0.85355339],
               [ 0.14644661],
               [-0.5       ]])
        """
    strike = np.radians(strike)
    dip = np.radians(dip)
    rake = np.radians(rake)
    ccw_to_x_stk = np.pi / 2 - strike
    ccw_to_x_dip = - strike
    z = np.sin(dip)
    l = np.cos(dip)
    x = np.cos(ccw_to_x_dip) * l
    y = np.sin(ccw_to_x_dip) * l
    rake_90 = np.array([x, y, -z])
    rake_0 = np.array([np.cos(ccw_to_x_stk), np.sin(ccw_to_x_stk), 0])
    return (np.sin(rake) * rake_90 + np.cos(rake) * rake_0).reshape(-1, 1)


def get_array(data, x, y):
    ar = xr.DataArray(
        data=data,
        dims=['y', 'x'],
        coords=dict(
            x=(['x'], x),
            y=(['y'], y)
        )
    )
    return ar