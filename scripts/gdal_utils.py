from osgeo import gdal, osr, ogr
import numpy as np




def arrray2wgstiff(path, array, lower_left_lon, lower_left_lat, x_dd, y_dd):
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(path, array.shape[1], array.shape[0], 1, gdal.GDT_Float32)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    print(lower_left_lon, x_dd, 0.0, lower_left_lat + y_dd *array.shape[0], 0.0, -y_dd)
    ds.SetGeoTransform((lower_left_lon, x_dd, 0.0, lower_left_lat + y_dd *array.shape[0], 0.0, -y_dd))
    ds.GetRasterBand(1).WriteArray(np.flipud(array))
    ds = None


def normal(strike, dip):
        nz = np.cos(dip)
        l = np.sin(dip)
        nx = np.cos(strike + np.pi / 2) * l
        ny = np.sin(strike + np.pi / 2) * l
        return np.array([nx, ny, nz]).reshape(-1, 1)

def shear_hat(strike, dip, rake):
    l_s = np.cos(rake)
    l_d = np.sin(rake)
    nz = -np.sin(dip) * l_d
    l = np.cos(dip) * l_d
    x_s = np.cos(strike) * l_s
    y_s = np.sin(strike) * l_s

    x_d = np.cos(strike + np.pi / 2) * l
    y_d = np.sin(strike + np.pi / 2) * l
    return np.array([x_s + x_d, y_s + y_d, nz]).reshape(-1, 1)