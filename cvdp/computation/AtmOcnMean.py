#!/usr/bin/env python3
"""
AtmOcnMean.py

CVDP functions for calculating means, standard deviations, and trends.
License: MIT
"""

import old_utils.analysis as an
from diag import compute_seasonal_avgs, compute_seasonal_trends
import xskillscore as xs
import numpy as np

nh_vars = ["NAM"]
sh_vars = ["SAM", "PSA1", "PSA2"]
eof_vars = nh_vars+sh_vars
latlon_dict = {}
latlon_dict['e'] = 0
latlon_dict['w'] = 360
# Set number of EOF functions(?)
eof_nums = 3
ptype = "trends"

def mean_seasonal_calc(ref_dataset, sim_dataset):

    print("\nCalculating climatological seasonal means...")
    ref_seas_avgs = compute_seasonal_avgs(ref_dataset)
    if "member" in ref_seas_avgs.coords:
        attrs = ref_seas_avgs.attrs  # save before doing groupby/mean
        members = ref_seas_avgs.member
        ref_seas_avgs = ref_seas_avgs.mean(dim="member")
        ref_seas_avgs.attrs = attrs
        ref_seas_avgs.attrs["members"] = members
    sim_seas_avgs = compute_seasonal_avgs(sim_dataset)
    if "member" in sim_seas_avgs.coords:
        attrs = sim_seas_avgs.attrs  # save before doing groupby/mean
        members = sim_seas_avgs.member
        sim_seas_avgs = sim_seas_avgs.mean(dim="member")
        sim_seas_avgs.attrs = attrs
        sim_seas_avgs.attrs["members"] = members

    #print("AHHHH",sim_seas_avgs,"\n\n")
    #sim_seas_avgs.to_netcdf("sim_seas_avgs.nc")
    #ref_seas_avgs.to_netcdf("ref_seas_avgs.nc")
    # If the cases are different shapes, we need to interpolate one to the other first
    #NOTE: the value that comes out of interp_diff is either None, or interpolated difference array
    arr_prime = an.interp_diff(sim_seas_avgs, ref_seas_avgs)

    # If arr_prime is None, then the two runs have already been interpolated (TS -> SST) or are the same grid/shape
    if arr_prime is None:
        arr_diff = sim_seas_avgs - ref_seas_avgs
    else:
        arr_diff = (arr_prime - ref_seas_avgs)

    return ref_seas_avgs, sim_seas_avgs, arr_diff

def trend_seasonal_calc(ref_dataset, sim_dataset):
    ref_seas_trnds = compute_seasonal_trends(ref_dataset)
    if "member" in ref_seas_trnds.coords:
        attrs = ref_seas_trnds.attrs  # save before doing groupby/mean
        members = ref_seas_trnds.member
        ref_seas_trnds = ref_seas_trnds.mean(dim="member")
        ref_seas_trnds.attrs = attrs
        ref_seas_trnds.attrs["members"] = members

    sim_seas_trnds = compute_seasonal_trends(sim_dataset)
    if "member" in sim_seas_trnds.coords:
        attrs = sim_seas_trnds.attrs  # save before doing groupby/mean
        members = sim_seas_trnds.member
        sim_seas_trnds = sim_seas_trnds.mean(dim="member")
        sim_seas_trnds.attrs = attrs
        sim_seas_trnds.attrs["members"] = members

    #sim_seas_trnds.to_netcdf("sim_seas_trnds.nc")
    #ref_seas_trnds.to_netcdf("ref_seas_trnds.nc")
    print("AtmOcnMean BEFORE EOF ref_seas_trnds",ref_seas_trnds,"\n")

    sim_attrs = sim_seas_trnds.attrs  # save before doing groupby/mean
    ref_attrs = ref_seas_trnds.attrs  # save before doing groupby/mean
    #for var in eof_vars:
    if 2==1:
        var = "SAM"
        if var in nh_vars:
            latlon_dict['n'] = 90
            latlon_dict['s'] = 20

        if var in sh_vars:
            latlon_dict['n'] = -20
            latlon_dict['s'] = -90

        # Get EOF
        #for i,arr in enumerate(finarrs):
        if 2==1:
            # Set EOF number for variable
            if var == "NAM" or var == "SAM":
                num = 0
            if var == "PSA1":
                num = 1
            if var == "PSA2":
                num = 2
            #print("sim_seas_trnds",sim_seas_trnds)
            eofs, pcs, SLP = an.get_eof(sim_seas_trnds, "SON", latlon_dict, eof_nums)
            pcs_num = pcs.sel(pc=num)
            pcs_norm_num = (pcs_num - pcs_num.mean(dim='time'))/pcs_num.std(dim='time')
            sim_seas_trnds = xs.linslope(pcs_norm_num, SLP, dim='time')
            sim_seas_trnds.attrs = sim_attrs

            print("AtmOcnMean sim_seas_trnds",sim_seas_trnds,"\n")

    if 2==1:
        var = "SAM"
        if var in nh_vars:
            latlon_dict['n'] = 90
            latlon_dict['s'] = 20

        if var in sh_vars:
            latlon_dict['n'] = -20
            latlon_dict['s'] = -90

        # Get EOF
        #for i,arr in enumerate(finarrs):
        if 2==1:
            # Set EOF number for variable
            if var == "NAM" or var == "SAM":
                num = 0
            if var == "PSA1":
                num = 1
            if var == "PSA2":
                num = 2
            #print(arr)
            print("num before EOF:", num)
            eofs, pcs, SLP = an.get_eof(ref_seas_trnds, "SON", latlon_dict, eof_nums)
            pcs_num = pcs.sel(pc=num)
            pcs_norm_num = (pcs_num - pcs_num.mean(dim='time'))/pcs_num.std(dim='time')
            ref_seas_trnds = xs.linslope(pcs_norm_num, SLP, dim='time')*-1
            ref_seas_trnds.attrs = ref_attrs

            print("AtmOcnMean ref_seas_trnds",ref_seas_trnds,"\n")
    
    season = "NDJFM"
    if season == "NDJFM":
        print("Here comes the fun cooker!")
        attrs = sim_seas_trnds.attrs  # Save metadata
        npi_ndjfm = sim_seas_trnds.sel(season=season).sel(lat=slice(30,65), lon=slice(160,220))
        npi_ndjfm = npi_ndjfm.weighted(np.cos(np.radians(npi_ndjfm.lat))).mean(dim=('lat','lon'))
        npi_ndjfm_standarized = (npi_ndjfm - npi_ndjfm.mean(dim='year'))/npi_ndjfm.std(dim='year')
        sim_seas_trnds = xs.linslope(npi_ndjfm_standarized, sim_seas_trnds.sel(season=season), dim='year')
        sim_seas_trnds.attrs = attrs
        #sim_seas_trnds = sim_seas_trnds.isel(year=0)

        attrs = ref_seas_trnds.attrs  # Save metadata
        npi_ndjfm = ref_seas_trnds.sel(season=season).sel(lat=slice(30,65), lon=slice(160,220))
        npi_ndjfm = npi_ndjfm.weighted(np.cos(np.radians(npi_ndjfm.lat))).mean(dim=('lat','lon'))
        npi_ndjfm_standarized = (npi_ndjfm - npi_ndjfm.mean(dim='year'))/npi_ndjfm.std(dim='year')
        ref_seas_trnds = xs.linslope(npi_ndjfm_standarized, ref_seas_trnds.sel(season=season), dim='year')
        ref_seas_trnds.attrs = attrs
        #ref_seas_trnds = ref_seas_trnds.isel(year=0)
        #arrs.append(npi)
        print("WTH IS HAPPENING",ref_seas_trnds)
        ref_seas_trnds.to_netcdf("ref_seas_trnds.nc")
    # If the cases are different shapes, we need to interpolate one to the other first
    #NOTE: the value that comes out of interp_diff is either None, or interpolated difference array
    arr_prime = an.interp_diff(sim_seas_trnds, ref_seas_trnds)

    # If arr_prime is None, then the two runs have already been interpolated (TS -> SST) or are the same grid/shape
    if arr_prime is None:
        arr_diff = sim_seas_trnds - ref_seas_trnds
    else:
        arr_diff = (arr_prime - ref_seas_trnds)

    return ref_seas_trnds, sim_seas_trnds, arr_diff