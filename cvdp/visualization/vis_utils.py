
import numpy as np
import xarray as xr
from matplotlib.colors import Normalize

# Land mask: for TS -> SST masking
def land_mask():
    """
    Mask land over TS variable to simulate SST's

    Takes premade land/sea classification netCDF file
        - 0: sea?
        - 1: land?
        - 2: ice?
        - 3: lakes?

    returns
    -------
     - ncl_masks: xarray.DataSet
        data set of the 4 classifications

     - lsmask: xarray.DataArray
        data array of the masked data set
    """
    ncl_masks = xr.open_mfdataset("landsea.nc", decode_times=True)
    lsmask = ncl_masks.LSMASK
    ncl_masks.close()
    return lsmask, ncl_masks


def clean_data(vn, wrap_data, ptype, diff=False):
    if diff:
        if vn == "ts":
            if ptype == "spatialmean":
                wrap_data = np.where(wrap_data<-6, -6, wrap_data)
    
        if vn == "psl":
            if ptype == "trends":
                wrap_data = np.where(wrap_data<-9, -9, wrap_data)
                wrap_data = np.where(wrap_data>9, 9, wrap_data)
            else:
                wrap_data = np.where(wrap_data<-11, -11, wrap_data)
    else:
        if vn == "ts":
            if ptype == "spatialmean":    
                wrap_data = np.where(wrap_data>40, np.nan, wrap_data)
                wrap_data = np.where(wrap_data<-6, -6, wrap_data)
    
        if vn == "psl":
            if ptype == "trends":
                wrap_data = np.where(wrap_data < -9, -9, wrap_data)
    return wrap_data


# Normailize colorbar for Rank plots
#-----------------------------------
class PiecewiseNorm(Normalize):
    def __init__(self, levels, clip=False):
        # the input levels
        self._levels = np.sort(levels)
        # corresponding normalized values between 0 and 1
        self._normed = np.linspace(0, 1, len(levels))
        Normalize.__init__(self, None, None, clip)

    def __call__(self, value, clip=None):
        # linearly interpolate to get the normalized value
        return np.ma.masked_array(np.interp(value, self._levels, self._normed))


