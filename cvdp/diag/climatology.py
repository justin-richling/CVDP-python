#!/usr/bin/env python3
"""
climatology.py

CVDP functions for calculating climatological means and standard deviations.
License: MIT
"""
import xarray as xr
import numpy as np


CLIMATOLOGY_SEASON_MONTHS = {
    "DJF": [12, 1, 2],
    "JFM": [1, 2, 3],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "JAS": [7, 8, 9],
    "SON": [9, 10, 11],
    "NDJFM": [11, 12, 1, 2, 3],
    "ANN": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
}


'''def compute_seasonal_avgs(var_data: xarray.DataArray, seasons: dict=CLIMATOLOGY_SEASON_MONTHS) -> xarray.DataArray:
    """
    Computes the sesonal averages for a given time series variable.
    
    :param var_data: Variable DataArray to compute the seasonal averages for.
    :type var_data: xarray.DataArray
    :param seasons: (Optional) Dictionary that maps the seasonal code (key) to its respective month integers (values)
    :type seasons: dict
    :return: Variable DataArray with the 'time' dimension reduced to seasons and their average values.
    :rtype: xarray.DataArray
    """

    """
    monthly_avgs = var_data.groupby("time.month").mean().rename(f"{var_data.name}_avg")
    seasonal_avgs = []
    for season_label in CLIMATOLOGY_SEASON_MONTHS:
        season_months = CLIMATOLOGY_SEASON_MONTHS[season_label]
        seasonal_avgs.append(monthly_avgs.sel(month=season_months).mean(dim="month"))
    return xarray.concat(seasonal_avgs, dim=xarray.DataArray(data=list(CLIMATOLOGY_SEASON_MONTHS.keys()), dims=["season"]))"""


    attrs = var_data.attrs  # save before doing groupby/mean

    # your existing logic
    monthly_avgs = var_data.groupby("time.month").mean().rename(f"{var_data.name}_avg")
    seasonal_avgs = []
    for season_label in CLIMATOLOGY_SEASON_MONTHS:
        season_months = CLIMATOLOGY_SEASON_MONTHS[season_label]
        seasonal_avg = monthly_avgs.sel(month=season_months).mean(dim="month")
        seasonal_avgs.append(seasonal_avg)

    # concat and restore attrs
    season_dim = xarray.DataArray(data=list(CLIMATOLOGY_SEASON_MONTHS.keys()), dims=["season"])
    seasonal_clim = xarray.concat(seasonal_avgs, dim=season_dim)

    # optional: name and attrs
    seasonal_clim.name = f"{var_data.name}_seasonal_clim_avg"
    seasonal_clim.attrs = attrs
    print("seasonal_clim:",seasonal_clim,"\n\n")

    return seasonal_clim'''



import cvdp_utils.avg_functions as af
season_dict = {"NDJFM":0,
               "DJF":0,
               "JFM":1,
               "MAM":3,
               "JJA":6,
               "JAS":7,
               "SON":9
}

#def compute_seasonal_avgs(arr, arr_anom, var_name, run_name) -> xr.DataArray:
def compute_seasonal_avgs(arr, var_name) -> xr.DataArray:


    # remove annual trend
    #--------------------
    farr_clim = arr.groupby('time.month').mean(dim='time')   # calculate climatology
    #farr_clim.attrs.update(farr.attrs)

    farr_anom = arr.groupby('time.month') - farr_clim   # form anomalies
    print("farr_anom:",farr_anom,"\n")
    farr_anom.attrs.update(arr.attrs)
    #farr_anom.attrs['run'] = run_name

    # Add desired start and end years to metadata
    season_yrs = np.unique(arr["time.year"])
    farr_anom.attrs['yrs'] = [season_yrs[0],season_yrs[-1]]

    # Rename variable to CVDP variable name
    farr_anom = farr_anom.rename(var_name)
    #print("fno_anom_loc is where farr_anom goes!",fno_anom_loc)
    #Path(fno_anom_loc).unlink(missing_ok=True)
    #farr_anom.to_netcdf(fno_anom_loc)



    units = arr.units
    season_yrs = np.unique(arr["time.year"])
    # Spatial Mean
    #-------------
    trnd_dict = {}
    ptype = "spatialmean"
    arr3 = arr.rolling(time=3, center=True).mean()
    arr3 = arr3.ffill(dim='time').bfill(dim='time').compute()
    run_name = arr.run_name

    arrANN = af.weighted_temporal_mean(arr)#.mean(dim='time')
    lintrndANN = arrANN.rename(var_name+'_ann')
    lintrndANN.attrs = {'units':units,'long_name':var_name+' (annual)','run':run_name,
                             'yrs':[season_yrs[0],season_yrs[-1]]}
    timefix = np.arange(season_yrs[0],season_yrs[-1]+1,1)
    lintrndANN["time"] = timefix
    trnd_dict[f'{var_name}_{ptype}_ann'] = lintrndANN

    for s in season_dict:
        lintrnd_da = arr3.isel( time=slice(season_dict[s], None, 12) )
        lintrnd_da = af.make_seasonal_da(var_name, run_name, lintrnd_da, units, s, season_yrs, ptype)
        trnd_dict[f'{var_name}_{ptype}_{s.lower()}'] = lintrnd_da
    if var_name == "psl":
        s = 'NDJFM'
        arr5 = arr.rolling(time=5, center=True).mean()
        arr5 = arr5.ffill(dim='time').bfill(dim='time').compute()
        lintrnd_da = arr5.isel( time=slice(season_dict[s], None, 12) )
        lintrnd_da = af.make_seasonal_da(var_name, run_name, lintrnd_da, units, s, season_yrs, ptype)
        trnd_dict[f'{var_name}_{ptype}_ndjfm'] = lintrnd_da

    """# Anomolies
    #----------
    #trnd_dict = {}
    ptype = "trends"
    # setup 3-month averages
    arr_anom3 = arr_anom.rolling(time=3, center=True).mean()
    arr_anom3 = arr_anom3.ffill(dim='time').bfill(dim='time').compute()

    arrANN_anom = af.weighted_temporal_mean(arr_anom)#.mean(dim='time')
    lintrndANN_anom = arrANN_anom.rename(var_name+'_ann')
    lintrndANN_anom.attrs = {'units':units,'long_name':var_name+' (annual)','run':run_name,
                             'yrs':[season_yrs[0],season_yrs[-1]]}
    #lintrndANN = lin_regress(lintrndANN)
    timefix = np.arange(season_yrs[0],season_yrs[-1]+1,1)
    lintrndANN_anom["time"] = timefix
    trnd_dict[f'{var_name}_{ptype}_ann'] = lintrndANN_anom

    for s in season_dict:
        lintrnd_da = arr_anom3.isel( time=slice(season_dict[s], None, 12) )
        lintrnd_da = af.make_seasonal_da(var_name, run_name, lintrnd_da, units, s, season_yrs, ptype)
        #lintrnd_da = lin_regress(lintrnd_da)
        lintrnd_da = lintrnd_da.drop_vars('month')
        trnd_dict[f'{var_name}_{ptype}_{s.lower()}'] = lintrnd_da
    if var_name == "psl":
        s = 'NDJFM'
        arr5 = arr_anom.rolling(time=5, center=True).mean()
        arr5 = arr5.ffill(dim='time').bfill(dim='time').compute()
        lintrnd_da = arr5.isel( time=slice(season_dict[s], None, 12) )

        lintrnd_da = af.make_seasonal_da(var_name, run_name, lintrnd_da, units, s, season_yrs, ptype)
        #lintrnd_da = lin_regress(lintrnd_da)
        lintrnd_da = lintrnd_da.drop_vars('month')
        trnd_dict[f'{var_name}_trends_ndjfm'] = lintrnd_da
    """

    #print(trnd_dict)

    ds = xr.Dataset(trnd_dict)
    ds.attrs['units']=units
    ds.attrs['run']=run_name
    ds.attrs['yrs']=[season_yrs[0],season_yrs[-1]]


    #arrDJF_anom, res, fit = lin_regress(arrDJF_anom)

    print("seasonal climo dataset:",ds,"\n\n")
    return ds









def compute_seasonal_stds(var_data: xarray.DataArray, seasons: dict=CLIMATOLOGY_SEASON_MONTHS) -> xarray.DataArray:
    monthly_avgs = var_data.groupby("time.month").mean().rename(f"{var_data.name}_std")
    seasonal_avgs = []
    for season_label in CLIMATOLOGY_SEASON_MONTHS:
        season_months = CLIMATOLOGY_SEASON_MONTHS[season_label]
        seasonal_avgs.append(monthly_avgs.sel(month=season_months).std(dim="month"))
    return xarray.concat(seasonal_avgs, dim=xarray.DataArray(data=list(CLIMATOLOGY_SEASON_MONTHS.keys()), dims=["season"]))


def compute_seasonal_trends(var_data: xarray.DataArray, seasons: dict=CLIMATOLOGY_SEASON_MONTHS) -> xarray.DataArray:
    monthly_avgs = var_data.groupby("time.month").mean()
    monthly_trend_avgs = (var_data.groupby("time.month") - monthly_avgs).rename(f"{var_data.name}_avg") # form anomalies
    seasonal_avgs = []
    for season_label in CLIMATOLOGY_SEASON_MONTHS:
        season_months = CLIMATOLOGY_SEASON_MONTHS[season_label]
        seasonal_avgs.append(monthly_trend_avgs.sel(month=season_months).mean(dim="month"))
    return xarray.concat(seasonal_avgs, dim=xarray.DataArray(data=list(CLIMATOLOGY_SEASON_MONTHS.keys()), dims=["season"]))