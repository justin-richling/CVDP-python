#!/usr/bin/env python3
"""
AtmOcnGR.py

CVDP functions for plotting means, standard deviations, trends, timeseries, metrics tables, etc.
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import xskillscore as xs

from vis.global_plots import (
    global_ensemble_plot,
    global_indmem_latlon_plot,
    global_indmemdiff_latlon_plot
)

from vis.polar_plots import (
    polar_ensemble_plot,
    polar_indmem_latlon_plot,
    polar_indmemdiff_latlon_plot
)

from vis.timeseries_plot import timeseries_plot
import cvdp_utils.avg_functions as af
import cvdp_utils.utils as helper_utils
import cvdp_utils.analysis as an

SEASON_LIST = ["DJF"]
VAR_SEASONS = {
    "psl": {"global": SEASON_LIST + ["NDJFM"], "polar": SEASON_LIST, "timeseries": SEASON_LIST},
    "ts": SEASON_LIST,
    "trefht": SEASON_LIST,
    "prect": SEASON_LIST,
}

EOF_VARS = ["NAM", "SAM", "PSA1", "PSA2"]
NH_VARS = ["NAM"]
SH_VARS = ["SAM", "PSA1", "PSA2"]

PTYPES = ["trends"]
MAP_TYPES = ["global", "polar", "timeseries"]
PLOT_TYPES = ["summary", "indmem", "indmemdiff"]


def get_plot_title(var, plot_type, ptype, season):
    base = f"{var} {ptype.capitalize()} ({season})"
    titles = {
        "summary": f"Ensemble Summary: {base}",
        "indmem": f"{base}\n",
        "indmemdiff": f"{base} Differences\n",
    }
    return titles.get(plot_type, "Unknown Title")


def get_plot_name(vn, var, ptype, season, plot_type, map_type):
    season_lower = season.lower()
    if ptype == "trends" and (var == "NPI" or var in EOF_VARS):
        suffix = f"pattern_{season_lower}"
    else:
        suffix = f"{ptype}_{season_lower}"
    plot_name = f"{vn}_{suffix}.{plot_type}.png"
    #if var == "NPI" or var in EOF_VARS:
    #    plot_name = f"{var.lower()}_pattern_{season_lower}.{plot_type}.png"
    
    #if map_type == "timeseries":
    #    plot_name = f"{var.lower()}_timeseries_{season_lower}.{plot_type}.png"
    
    #if 1==2:
    #    suffix = f"pattern_{season_lower}" if ptype == "trends" else f"{ptype}_{season_lower}"
    #    plot_name = f"{vn}_{suffix}.{plot_type}.png"
    return plot_name


def compute_trend(data):
    return af.lin_regress(data)[0]


def compute_diff(sim, ref):
    interp = an.interp_diff(sim, ref)
    return sim - ref if interp is None else sim - interp


def compute_npi(sim_ts, ref_ts):
    def _standardize(arr):
        return (arr - arr.mean("time")) / arr.std("time")

    def _npi(arr):
        sliced = arr.sel(lat=slice(30, 65), lon=slice(160, 220))
        weighted = sliced.weighted(np.cos(np.radians(sliced.lat))).mean(["lat", "lon"])
        return _standardize(weighted)

    sim_npi = xs.linslope(_npi(sim_ts), sim_ts, dim="time")
    ref_npi = xs.linslope(_npi(ref_ts), ref_ts, dim="time")
    return sim_npi, ref_npi, compute_diff(sim_npi, ref_npi)


def compute_eof(var, sim_anom, ref_anom, season):
    #num = EOF_VARS.index(var)
    if var == "NAM" or var == "SAM":
        num = 0
    if var == "PSA1":
        num = 1
    if var == "PSA2":
        num = 2
    bounds = {'n': 90, 's': 20} if var in NH_VARS else {'n': -20, 's': -90}

    def _eof(arr, invert=False):
        eofs, pcs, slp = an.get_eof(arr, season, bounds, neof=3)
        pcs_std = (pcs.sel(pc=num) - pcs.sel(pc=num).mean("time")) / pcs.sel(pc=num).std("time")
        return xs.linslope(-pcs_std if invert else pcs_std, slp, dim="time"), pcs_std

    sim_pattern, sim_pc = _eof(sim_anom, var == "SAM")
    ref_pattern, ref_pc = _eof(ref_anom, var == "PSA2")
    return sim_pattern, ref_pattern, compute_diff(sim_pattern, ref_pattern), sim_pc, ref_pc


def plot_dispatch(plot_type, ptype, map_type, vn, var, sim, ref, diff, vtres, title, pcs=None):
    if plot_type == "summary":
        if map_type == "polar":
            return polar_ensemble_plot([sim, ref], diff, vn, var, ptype, vtres, title)
        if map_type == "global":
            return global_ensemble_plot([sim, ref], diff, vn, ptype, vtres, title)
        if map_type == "timeseries" and pcs:
            return timeseries_plot(var, pcs[0], pcs[1])
    elif plot_type == "indmem":
        if map_type == "polar":
            return polar_indmem_latlon_plot(vn, var, [sim, ref], vtres, title, ptype)
        if map_type == "global":
            return global_indmem_latlon_plot(vn, [sim, ref], vtres, title, ptype)
        if map_type == "timeseries" and pcs:
            return timeseries_plot(var, pcs[0], pcs[1])
    elif plot_type == "indmemdiff":
        run = f"{sim.run.values} - {ref.run.values}"
        if map_type == "polar":
            return polar_indmemdiff_latlon_plot(vn, var, run, diff, ptype, vtres, title)
        if map_type == "global":
            return global_indmemdiff_latlon_plot(vn, run, diff, ptype, vtres, title)
    return None


def graphics(plot_loc, **kwargs):
    res = helper_utils.get_variable_defaults()
    vn = kwargs["vn"]

    for ptype in PTYPES:
        for map_type in MAP_TYPES:
            vres = res[vn][ptype]
            seasons = VAR_SEASONS[vn][map_type] if isinstance(VAR_SEASONS[vn], dict) else VAR_SEASONS[vn]

            for season in seasons:
                for plot_type in PLOT_TYPES:
                    key = f"{vn}_{ptype}_{season.lower()}"
                    sim_data = kwargs["sim_seas"][key]
                    ref_data = kwargs["ref_seas"][key]
                    figs = []
                    print(vn, plot_type, ptype, map_type, season)
                    if ptype == "trends" and vn == "psl" and map_type == "global" and season == "NDJFM":
                        sim_npi, ref_npi, diff_npi = compute_npi(kwargs["sim_seas_ts"][key], kwargs["ref_seas_ts"][key])
                        title = get_plot_title("NPI", plot_type, ptype, season)
                        name = get_plot_name(vn, "NPI", ptype, season, plot_type, map_type)
                        print("fig name:",name)
                        fig = plot_dispatch(plot_type, ptype, map_type, vn, "NPI", sim_npi, ref_npi, diff_npi, vres, title)
                        if fig: figs.append((fig, name))

                    elif ptype == "trends" and vn == "psl" and map_type in ["polar", "timeseries"]:
                        for var in EOF_VARS:
                            print("\t",var,"\n")
                            sim, ref, diff, sim_pc, ref_pc = compute_eof(var, kwargs["sim_season_anom_avgs"], kwargs["ref_season_anom_avgs"], season)
                            title = get_plot_title(var, plot_type, ptype, season)
                            name = get_plot_name(vn, var, ptype, season, plot_type, map_type)
                            fig = plot_dispatch(plot_type, ptype, map_type, vn, var, sim, ref, diff, vres, title, pcs=(sim_pc, ref_pc))
                            print(name)
                            if fig: figs.append((fig, name))
                    else:
                        if season != "NDJFM":
                            sim = compute_trend(sim_data) if ptype == "trends" else sim_data.mean("time")
                            ref = compute_trend(ref_data) if ptype == "trends" else ref_data.mean("time")
                            diff = compute_diff(sim, ref)
                            title = get_plot_title(vn.upper(), plot_type, ptype, season)
                            name = get_plot_name(vn, vn, ptype, season, plot_type, map_type)
                            print("fig name:",name)
                            fig = plot_dispatch(plot_type, ptype, map_type, vn, vn, sim, ref, diff, vres, title)
                            if fig: figs.append((fig, name))

                    for fig, name in figs:
                        fig.savefig(plot_loc / name, bbox_inches="tight")
                        plt.close(fig)
