#!/usr/bin/env python3
"""
AtmOcnMean.py

CVDP functions for calculating means, standard deviations, and trends.
License: MIT
"""

import cvdp_utils.analysis as an
from diag import compute_seasonal_avgs
from pathlib import Path
import xarray as xr

def mean_seasonal_calc(ds_name, dataset, var_name, config_dict):
    save_loc = config_dict[ds_name]["save_loc"]#.mkdir(parents=True, exist_ok=True)
    syr = config_dict[ds_name]["syr"]
    eyr = config_dict[ds_name]["eyr"]

    avgs_filname = f'{ds_name}.cvdp_data.{var_name}.climo.avgs.{syr}-{eyr}.nc'
    anom_avgs_filename = f'{ds_name}.cvdp_data.{var_name}.climo.anom_avgs.{syr}-{eyr}.nc'
    ts_filename = f'{ds_name}.cvdp_data.{var_name}.climo.ts.{syr}-{eyr}.nc'
    avgs_fno = Path(avgs_filname)
    anom_avgs_fno = Path(anom_avgs_filename)
    ts_fno = Path(ts_filename)
    if avgs_fno.is_file() and anom_avgs_fno.is_file() and ts_fno.is_file():
        print(f"\nFound pre-existing climatology files for {ds_name} {var_name}, loading from disk...\n")
        seas_avgs = xr.open_dataarray(save_loc / avgs_fno)
        season_anom_avgs = xr.open_dataarray(save_loc / anom_avgs_fno)
        seas_ts = xr.open_dataarray(save_loc / ts_fno)
        data_dict = {
            "seas_avgs": seas_avgs,
            "season_anom_avgs": season_anom_avgs,
            "seas_ts": seas_ts,
        }
        return data_dict
    print("\nCalculating climatological seasonal means...")
    seas_avgs, season_anom_avgs, seas_ts = compute_seasonal_avgs(dataset, var_name)
    if "member" in seas_avgs.coords:
        attrs = seas_avgs.attrs  # save before doing groupby/mean
        members = seas_avgs.member
        seas_avgs = seas_avgs.mean(dim="member")
        seas_avgs.attrs = attrs
        seas_avgs.attrs["members"] = members
    
    #ds = xr.Dataset(trnd_dict)
    #ds = ds.assign_coords(run=run_name, units=units, syr=syr, eyr=eyr)
    #for ds_name in config["Data"]:
        #syr, eyr = config["Data"][ds_name]["start_yr"], config["Data"][ds_name]["end_yr"]
        #save_loc = Path( config["Paths"]["nc_save_loc"] )
        #save_loc.mkdir(parents=True, exist_ok=True)

    #fno = f'{ds_name}.cvdp_data.{var_name}.climo.avgs.{syr}-{eyr}.nc'
    file_name = save_loc / avgs_fno
    seas_avgs.to_netcdf(file_name)

    #fno = f'{ds_name}.cvdp_data.{var_name}.climo.anom_avgs.{syr}-{eyr}.nc'
    file_name = save_loc / anom_avgs_fno
    season_anom_avgs.to_netcdf(file_name)

    #fno = f'{ds_name}.cvdp_data.{var_name}.climo.ts.{syr}-{eyr}.nc'
    file_name = save_loc / ts_fno
    seas_ts.to_netcdf(file_name)

    #return ref_seas_avgs, sim_seas_avgs, ref_season_anom_avgs, sim_season_anom_avgs, ref_seas_ts, sim_seas_ts
    data_dict = {
        "seas_avgs": seas_avgs,
        "season_anom_avgs": season_anom_avgs,
        "seas_ts": seas_ts,
    }
    return data_dict


'''
def mean_seasonal_calc(ref_name, ref_dataset, sim_name, sim_dataset, var_name, config_dict):

    print("\nCalculating climatological seasonal means...")
    ref_seas_avgs, ref_season_anom_avgs, ref_seas_ts = compute_seasonal_avgs(ref_dataset, var_name)
    if "member" in ref_seas_avgs.coords:
        attrs = ref_seas_avgs.attrs  # save before doing groupby/mean
        members = ref_seas_avgs.member
        ref_seas_avgs = ref_seas_avgs.mean(dim="member")
        ref_seas_avgs.attrs = attrs
        ref_seas_avgs.attrs["members"] = members
    sim_seas_avgs, sim_season_anom_avgs, sim_seas_ts = compute_seasonal_avgs(sim_dataset, var_name)
    if "member" in sim_seas_avgs.coords:
        attrs = sim_seas_avgs.attrs  # save before doing groupby/mean
        members = sim_seas_avgs.member
        sim_seas_avgs = sim_seas_avgs.mean(dim="member")
        sim_seas_avgs.attrs = attrs
        sim_seas_avgs.attrs["members"] = members
    #ds = xr.Dataset(trnd_dict)
    #ds = ds.assign_coords(run=run_name, units=units, syr=syr, eyr=eyr)
    #for ds_name in config["Data"]:
        #syr, eyr = config["Data"][ds_name]["start_yr"], config["Data"][ds_name]["end_yr"]
        #save_loc = Path( config["Paths"]["nc_save_loc"] )
        #save_loc.mkdir(parents=True, exist_ok=True)

    # Save climatological seasonal means to netCDF
    save_loc = config_dict[ref_name]["save_loc"]#.mkdir(parents=True, exist_ok=True)
    syr = config_dict[ref_name]["syr"]
    eyr = config_dict[ref_name]["eyr"]
    fno = f'{ref_name}.cvdp_data.{var_name}.climo.avgs.{syr}-{eyr}.nc'
    file_name = save_loc / fno
    ref_seas_avgs.to_netcdf(file_name)

    fno = f'{ref_name}.cvdp_data.{var_name}.climo.anom_avgs.{syr}-{eyr}.nc'
    file_name = save_loc / fno
    ref_season_anom_avgs.to_netcdf(file_name)

    fno = f'{ref_name}.cvdp_data.{var_name}.climo.ts.{syr}-{eyr}.nc'
    file_name = save_loc / fno
    ref_seas_ts.to_netcdf(file_name)



    save_loc = config_dict[ref_name]["save_loc"]#.mkdir(parents=True, exist_ok=True)
    syr = config_dict[ref_name]["syr"]
    eyr = config_dict[ref_name]["eyr"]
    fno = f'{ref_name}.cvdp_data.{var_name}.climo.avgs.{syr}-{eyr}.nc'
    file_name = save_loc / fno
    ref_seas_avgs.to_netcdf(file_name)

    fno = f'{ref_name}.cvdp_data.{var_name}.climo.anom_avgs.{syr}-{eyr}.nc'
    file_name = save_loc / fno
    ref_season_anom_avgs.to_netcdf(file_name)

    fno = f'{ref_name}.cvdp_data.{var_name}.climo.ts.{syr}-{eyr}.nc'
    file_name = save_loc / fno
    ref_seas_ts.to_netcdf(file_name)

    """# If the cases are different shapes, we need to interpolate one to the other first
    #NOTE: the value that comes out of interp_diff is either None, or interpolated difference array
    arr_prime = an.interp_diff(sim_seas_avgs, ref_seas_avgs)

    # If arr_prime is None, then the two runs have already been interpolated (TS -> SST) or are the same grid/shape
    if arr_prime is None:
        arr_diff = sim_seas_avgs - ref_seas_avgs
    else:
        arr_diff = (arr_prime - ref_seas_avgs)"""




    """print("\nCalculating climatological seasonal trends...")
    ref_seas_trends = compute_seasonal_trends(ref_dataset, var_name)
    if "member" in ref_seas_trends.coords:
        attrs = ref_seas_trends.attrs  # save before doing groupby/mean
        members = ref_seas_trends.member
        ref_seas_trends = ref_seas_trends.mean(dim="member")
        ref_seas_trends.attrs = attrs
        ref_seas_trends.attrs["members"] = members
    sim_seas_trends = compute_seasonal_trends(sim_dataset, var_name)
    if "member" in sim_seas_trends.coords:
        attrs = sim_seas_trends.attrs  # save before doing groupby/mean
        members = sim_seas_trends.member
        sim_seas_trends = sim_seas_trends.mean(dim="member")
        sim_seas_trends.attrs = attrs
        sim_seas_trends.attrs["members"] = members

    # If the cases are different shapes, we need to interpolate one to the other first
    #NOTE: the value that comes out of interp_diff is either None, or interpolated difference array
    arr_prime = an.interp_diff(sim_seas_trends, ref_seas_trends)

    # If arr_prime is None, then the two runs have already been interpolated (TS -> SST) or are the same grid/shape
    if arr_prime is None:
        arr_diff = sim_seas_trends - ref_seas_trends
    else:
        arr_diff = (arr_prime - ref_seas_trends)"""


    #sim_seas_avg, sim_res, sim_fit = af.lin_regress(sim_seas_avgs[f"{vn}_{type}_{season.lower()}"])
    #ref_seas_avg, ref_res, res_fit = af.lin_regress(ref_seas_avgs[f"{vn}_{type}_{season.lower()}"])
    
    #return ref_seas_avgs, sim_seas_avgs, ref_season_anom_avgs, sim_season_anom_avgs, ref_seas_ts, sim_seas_ts
    data_dict = {
        "ref_seas_avgs": ref_seas_avgs,
        "sim_seas_avgs": sim_seas_avgs,
        "ref_season_anom_avgs": ref_season_anom_avgs,
        "sim_season_anom_avgs": sim_season_anom_avgs,
        "ref_seas_ts": ref_seas_ts,
        "sim_seas_ts": sim_seas_ts
    }
    return data_dict
'''