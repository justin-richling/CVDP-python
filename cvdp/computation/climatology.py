#!/usr/bin/env python3
"""
climatology.py

CVDP functions for calculating climatological means and standard deviations.
License: MIT
"""
import xarray
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
    #print("monthly_avgs:",monthly_avgs,"\n\n")
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
    #print("seasonal_clim:",seasonal_clim,"\n\n")
    return seasonal_clim

'''




def compute_seasonal_avgs(var_data: xarray.DataArray, seasons: dict = CLIMATOLOGY_SEASON_MONTHS) -> xarray.DataArray:
    """
    Computes the seasonal averages for a given time series variable per year.

    :param var_data: Variable DataArray to compute the seasonal averages for.
    :type var_data: xarray.DataArray
    :param seasons: Dictionary mapping season names to month numbers
    :type seasons: dict
    :return: DataArray with 'season' and 'year' dimensions
    :rtype: xarray.DataArray
    """

    attrs = var_data.attrs  # Save original attributes

    # Add month and year as coordinates for easier grouping
    var_data = var_data.assign_coords(
        month=("time", var_data["time.month"].data),
        year=("time", var_data["time.year"].data)
    )

    seasonal_avgs = []
    years = sorted(set(var_data.year.values))

    for season_label, season_months in seasons.items():
        # Select months in season
        season_data = var_data.where(var_data.month.isin(season_months), drop=True)

        # Group by year and average over time
        season_avg = season_data.groupby("year").mean(dim="time")

        # Add 'season' dimension to result
        season_avg = season_avg.expand_dims(dim={"season": [season_label]})

        seasonal_avgs.append(season_avg)

    # Concatenate all seasons
    seasonal_clim = xarray.concat(seasonal_avgs, dim="season")

    # Optional: assign 'time' coordinate using mid-month timestamps (for plotting)
    mid_month = {s: int(np.median(m)) for s, m in seasons.items()}
    time_vals = [np.datetime64(f"{int(y)}-{mid_month[s]:02d}-15") for s in seasons for y in years]
    seasonal_clim = seasonal_clim.assign_coords(time=("season", time_vals[:len(seasonal_clim.season)]))

    # Set name and restore attrs
    seasonal_clim.name = f"{var_data.name}_seasonal_avg"
    seasonal_clim.attrs = attrs
    print("seasonal_clim:",seasonal_clim,"\n\n")
    return seasonal_clim







def compute_seasonal_stds(var_data: xarray.DataArray, seasons: dict=CLIMATOLOGY_SEASON_MONTHS) -> xarray.DataArray:
    monthly_avgs = var_data.groupby("time.month").mean().rename(f"{var_data.name}_std")
    seasonal_avgs = []
    for season_label in CLIMATOLOGY_SEASON_MONTHS:
        season_months = CLIMATOLOGY_SEASON_MONTHS[season_label]
        seasonal_avgs.append(monthly_avgs.sel(month=season_months).std(dim="month"))
    return xarray.concat(seasonal_avgs, dim=xarray.DataArray(data=list(CLIMATOLOGY_SEASON_MONTHS.keys()), dims=["season"]))

"""
climatology = ref_ds.psl.groupby("time.month").mean()
anom = ref_ds.psl.groupby("time.month") - climatology

# Fix the coordinate so 'month' is properly indexed along 'time'
anom = anom.assign_coords(month=("time", anom["month"].data))

# Now this works
monthly_anom = anom.groupby("month").mean()
monthly_anom.sel(month=CLIMATOLOGY_SEASON_MONTHS["JJA"])
"""

"""def compute_seasonal_trends(var_data: xarray.DataArray, seasons: dict=CLIMATOLOGY_SEASON_MONTHS) -> xarray.DataArray:
    monthly_avgs = var_data.groupby("time.month").mean()
    monthly_trend_avgs = (var_data.groupby("time.month") - monthly_avgs).rename(f"{var_data.name}_avg") # form anomalies
    monthly_trend_avgs = monthly_trend_avgs.assign_coords(month=("time", monthly_trend_avgs["month"].data))
    trend_avgs = []
    for season_label in seasons:
        season_months = seasons[season_label]
        monthly_anom = monthly_trend_avgs.groupby("month").mean()
        monthly_anom.sel(month=season_months)
        trend_avgs.append(monthly_anom)
        #trend_avgs.append(monthly_trend_avgs.sel(month=season_months).mean(dim="month"))
    return xarray.concat(trend_avgs, dim=xarray.DataArray(data=list(seasons.keys()), dims=["season"]))"""


'''def compute_seasonal_trends(var_data: xarray.DataArray, seasons: dict=CLIMATOLOGY_SEASON_MONTHS) -> xarray.DataArray:

    attrs = var_data.attrs  # save before doing groupby/mean
    var_data = var_data.assign_coords(month=('time', var_data['time.month'].data))

    farr_clim = var_data.groupby('time.month').mean(dim='time').rename(f"{var_data.name}_trend")   # calculate climatology
    farr_anom = var_data.groupby('month') - farr_clim
    #grouped = np.unique(farr_anom.time.dt.year)
    #farr_clim.attrs.update(farr.attrs)
    
    #farr_anom = var_data.groupby('time.month') - farr_clim   # form anomalies
    farr_anom.coords['month'] = ('time', var_data['time.month'].data)
    #farr_anom.coords['year'] = grouped

    # Reassign the 'month' coordinate explicitly if needed
    print("farr_anom:",farr_anom,"\n\n")
    seasonal_avgs = []
    for season_label in seasons:
        season_months = seasons[season_label]
        monthly_anom = farr_anom.groupby("month").mean()
        seasonal_avg = monthly_anom.sel(month=season_months)
        #seasonal_avg = seasonal_avg.assign_coords(time=("season", grouped))


        #grouped = seasonal_avg.groupby("year").mean(dim="time")
        #grouped = np.unique(seasonal_avg.time.dt.year)
        #seasonal_avg = farr_anom.sel(month=season_months).mean(dim="month")
        #seasonal_avg = farr_anom.where(farr_anom.month.isin(season_months), drop=True).mean(dim="month")
        #time_values = [np.datetime64(f"{y}") for y in grouped]
        #seasonal_avgs.extend(grouped)

        seasonal_avgs.append(seasonal_avg)
    
    # concat and restore attrs
    season_dim = xarray.DataArray(data=list(seasons.keys()), dims=["season"])
    seasonal_clim = xarray.concat(seasonal_avgs, dim=season_dim)

    # optional: name and attrs
    seasonal_clim.name = f"{var_data.name}_seasonal_clim_trend"
    seasonal_clim.attrs = attrs

    #print("farr_anom:",farr_anom,"\n")
    #farr_anom.attrs.update(var_data.attrs)
    print("seasonal_clim:",seasonal_clim,"\n\n")
    #return xarray.concat(farr_anom, dim="time")
    return seasonal_clim'''





def compute_seasonal_trends(var_data: xarray.DataArray, seasons: dict = CLIMATOLOGY_SEASON_MONTHS) -> xarray.DataArray:
    """
    Compute seasonal anomalies per year.

    :param var_data: Input data array with 'time' dimension.
    :param seasons: Dict mapping season labels (e.g. 'DJF') to lists of month integers.
    :return: DataArray with 'season' and 'year' dimensions and climatological anomalies.
    """

    attrs = var_data.attrs  # Save original attributes

    # Add 'month' and 'year' as coordinates
    var_data = var_data.assign_coords(
        month=("time", var_data["time.month"].data),
        year=("time", var_data["time.year"].data)
    )

    # Step 1: Compute monthly climatology
    monthly_clim = var_data.groupby("time.month").mean(dim="time").rename(f"{var_data.name}_clim")

    # Step 2: Subtract climatology to compute anomalies
    anomalies = var_data.groupby("month") - monthly_clim

    # Ensure 'month' and 'year' are still assigned after groupby subtraction
    anomalies = anomalies.assign_coords(
        month=("time", anomalies["time.month"].data),
        year=("time", anomalies["time.year"].data)
    )

    seasonal_avgs = []
    season_labels = []
    year_values = []

    for season_label, season_months in seasons.items():
        # Filter anomalies for the season's months
        season_anom = anomalies.where(anomalies.month.isin(season_months), drop=True)

        # Group by year and average over time
        season_avg = season_anom.groupby("year").mean(dim="time")

        # Add 'season' dimension
        season_avg = season_avg.expand_dims(season=[season_label])
        seasonal_avgs.append(season_avg)

        # Store year values for assigning time later
        year_values = season_avg.year.values  # Same for all seasons
        season_labels.append(season_label)

    # Combine all seasons
    seasonal_trends = xarray.concat(seasonal_avgs, dim="season")

    # Optional: assign 'time' coordinate (middle month of each season)
    mid_months = {s: int(np.median(m)) for s, m in seasons.items()}
    time_coord = [
        np.datetime64(f"{int(y)}-{mid_months[s]:02d}-15")
        for s in seasonal_trends.season.values
        for y in seasonal_trends.year.values
    ]
    time_coord = np.array(time_coord).reshape(len(seasonal_trends.season), len(seasonal_trends.year))

    seasonal_trends = seasonal_trends.assign_coords(time=(("season", "year"), time_coord))

    # Name and attrs
    seasonal_trends.name = f"{var_data.name}_seasonal_clim_trend"
    seasonal_trends.attrs = attrs
    print("seasonal_trends",seasonal_trends)
    return seasonal_trends

