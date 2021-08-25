import numpy as np

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
    """
    return dd * (111319.9 * np.cos(np.deg2rad(lat)))