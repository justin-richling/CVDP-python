#!/usr/bin/env python3
"""
AtmOcnGR.py

CVDP functions for plotting means, standard deviations, trends, timeseries, metrics tables, etc.
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from vis.global_plots import (
    global_ensemble_plot,
    global_indmem_latlon_plot,
    global_indmemdiff_latlon_plot
)
import cvdp_utils.avg_functions as af
import cvdp_utils.utils as helper_utils
import cvdp_utils.analysis as an
#import cvdp.cvdp_utils.utils as helper_utils

season_list = ["DJF", "JFM", "MAM", "JJA", "JAS", "SON", "ANN"]
var_seasons = {
    "psl": {"global": season_list + ["NDJFM"], "polar": season_list},
    "ts": season_list,
    "trefht": season_list,
    "prect": season_list,
}

ptypes = ["spatialmean", "trends"]
vns = ["psl"]
map_types = ["global"]
plot_types = ["summary", "indmem", "indmemdiff"]


def get_plot_name_and_title(vn, var, ptype, season, plot_type, map_type):
    season_upper = season.upper()
    season_lower = season.lower()

    base_var = "sst" if vn == "ts" else vn
    title_var = "SST" if vn == "ts" else vn.upper()

    if map_type == "global":
        if ptype == "trends" and vn == "psl" and season == "NDJFM":
            plot_name = f"npi_pattern_{season_lower}.{plot_type}.png"
            var = "NPI"
        else:
            suffix = f"{ptype}_{season_lower}.{plot_type}.png"
            plot_name = f"{base_var}_{suffix}" if ptype != "trends" else f"{base_var}_pattern_{season_lower}.{plot_type}.png"
    elif map_type == "polar":
        plot_name = f"{base_var}_pattern_{season_lower}.{plot_type}.png"
    else:
        plot_name = "unknown.png"

    title = {
        "summary": f"Ensemble Summary: {var or title_var} {ptype.capitalize()} ({season_upper})",
        "indmem": f"{var or title_var} {ptype.capitalize()} ({season_upper})\n",
        "indmemdiff": f"{var or title_var} {ptype.capitalize()} Differences ({season_upper})\n",
    }.get(plot_type, "Unknown Title")

    return plot_name, title


def compute_mean_diff(sim, ref):
    arr_prime = an.interp_diff(sim, ref)
    return sim - ref if arr_prime is None else arr_prime - ref


def compute_trend(data):
    return af.lin_regress(data)[0]  # returns the trend array


def handle_plot(plot_type, ptype, map_type, vn, season, vtres, sim_data, ref_data):
    sim = sim_data.mean(dim="time") if ptype == "spatialmean" else compute_trend(sim_data)
    ref = ref_data.mean(dim="time") if ptype == "spatialmean" else compute_trend(ref_data)
    diff = compute_mean_diff(sim, ref)

    var = "NPI" if vn == "psl" and season == "NDJFM" else vn
    plot_name, title = get_plot_name_and_title(vn, var, ptype, season, plot_type, map_type)

    if plot_type == "summary":
        fig = global_ensemble_plot([sim, ref], diff, vn, ptype, vtres, title)
    elif plot_type == "indmem":
        fig = global_indmem_latlon_plot(vn, [sim, ref], vtres, title, plot_type)
    elif plot_type == "indmemdiff":
        run = f"{sim.run.values} - {ref.run.values}"
        fig = global_indmemdiff_latlon_plot(vn, run, diff, plot_type, vtres, title)
    else:
        fig = None

    return fig, plot_name


def graphics(plot_loc, **kwargs):
    print("\nPlotting climatological seasonal means...")
    ref_seas_avgs = kwargs["ref_seas"]
    sim_seas_avgs = kwargs["sim_seas"]
    res = helper_utils.get_variable_defaults()
    vn = kwargs["vn"]

    for ptype in ptypes:
        for map_type in map_types:
            vres = res[vn]
            vtres = vres[ptype]
            seasons = var_seasons[vn][map_type] if isinstance(var_seasons[vn], dict) else var_seasons[vn]

            for season in seasons:
                for plot_type in plot_types:
                    key = f"{vn}_{ptype}_{season.lower()}"
                    sim_data = sim_seas_avgs[key]
                    ref_data = ref_seas_avgs[key]

                    fig, plot_name = handle_plot(plot_type, ptype, map_type, vn, season, vtres, sim_data, ref_data)

                    if fig:
                        fig.savefig(plot_loc / plot_name, bbox_inches="tight")
                        plt.close(fig)
