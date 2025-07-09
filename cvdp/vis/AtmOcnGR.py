#!/usr/bin/env python3
"""
AtmOcnGR.py

CVDP functions for plotting means, standard deviations, trends, timeseries, metrics tables, etc.
License: MIT
"""

import numpy as np
import vis as vis
#from visualization.global_plots import *
import matplotlib.pyplot as plt
import utils as helper_utils

print("helper_utils????",dir(helper_utils),"\n\n")

season_list = ["DJF", "JFM", "MAM", "JJA", "JAS", "SON", "ANN"]
var_seasons = {
    "psl": {"global": season_list, "polar": season_list + ["NDJFM"]},
    "ts": season_list,
    "trefht": season_list,
    "prect": season_list,
}

nh_vars = ["NAM"]
sh_vars = ["SAM", "PSA1", "PSA2"]
eof_vars = nh_vars + sh_vars

ptypes = ["spatialmean"]
vns = ["psl"]
map_types = ["global"]

def get_plot_name_and_title(vn, var, type, season, plot_type, map_type):
    season_upper = season.upper()
    season_lower = season.lower()

    if map_type == "global":
        base_var = "sst" if vn == "ts" else vn
        plot_name = f"{base_var}_{type}_{season_lower}.{plot_type}.png"
        if type == "trends":
            if vn == "psl":
                if season == "NDJFM":
                    plot_name = f"npi_pattern_{season_lower}.{plot_type}.png"
                    title_map = {
                        "summary": f"Ensemble Summary: NPI Pattern ({season_upper})",
                        "indmem": f"NPI Pattern ({season_upper})\n",
                        "indmemdiff": f"NPI Pattern Differences ({season_upper})\n",
                    }
                else:
                    title_map = {
                        "summary": f"Ensemble Summary: {var} Pattern ({season_upper})\n",
                        "indmem": f"{var} Pattern ({season_upper})\n",
                        "indmemdiff": f"{var} Pattern Differences ({season_upper})\n",
                    }
        else:
            title_map = {
                "summary": f"Ensemble Summary: {'SST' if vn == 'ts' else vn.upper()} {type.capitalize()} ({season_upper})",
                "indmem": f"{'SST' if vn == 'ts' else vn.upper()} {type.capitalize()} ({season_upper})\n",
                "indmemdiff": f"{'SST' if vn == 'ts' else vn.upper()} {type.capitalize()} Differences ({season_upper})\n",
            }

    elif map_type == "polar":
        if vn == "ts":
            plot_name = f"sst_{type}_{season_lower}.{plot_type}.png"
            title_map = {
                "summary": f"Ensemble Summary: SST {type.capitalize()} ({season_upper})",
                "indmem": f"SST {type.capitalize()} ({season_upper})\n",
                "indmemdiff": f"SST {type.capitalize()} Differences ({season_upper})\n",
            }
        if vn == "psl":
            plot_name = f"{var.lower()}_pattern_{season_lower}.{plot_type}.png"
            title_map = {
                "summary": f"Ensemble Summary: {var} Pattern ({season_upper})\n",
                "indmem": f"{var} Pattern ({season_upper})\n",
                "indmemdiff": f"{var} Pattern Differences ({season_upper})\n",
            }
        else:
            plot_name = f"{vn}_{type}_{season_lower}.{plot_type}.png"
            title_map = {
                "summary": f"Ensemble Summary: {vn.upper()} {type.capitalize()} ({season_upper})",
                "indmem": f"{vn.upper()} {type.capitalize()} ({season_upper})\n",
                "indmemdiff": f"{vn.upper()} {type.capitalize()} Differences ({season_upper})\n",
            }
    else:
        plot_name = "unknown.png"
        title_map = {"summary": "Unknown Title", "indmem": "", "indmemdiff": ""}

    print("\n\nplot_name, title_map[plot_type]",plot_name, title_map[plot_type],"\n\n")
    return plot_name, title_map[plot_type]


def graphics(plot_loc, **kwargs):
    print("\nPlotting climatological seasonal means...")
    ref_seas_avgs = kwargs["ref_seas"]
    sim_seas_avgs = kwargs["sim_seas"]
    seas_avgs_diff = kwargs["diff_seas"]
    res = helper_utils.utils.get_variable_defaults()

    for vn in vns:
        for type in ptypes:
            for map_type in map_types:
                vres = res[vn]
                vtres = vres[type]
                seasons = var_seasons[vn][map_type] if isinstance(var_seasons[vn], dict) else var_seasons[vn]

                for season in seasons:
                    for plot_type in ["summary", "indmem","indmemdiff"]:
                        plot_name, title = get_plot_name_and_title(vn, type, season, plot_type)

                        if plot_type == "summary":
                            if map_type == "global":
                                if type == "spatialmean":
                                    plot_name, title = get_plot_name_and_title(vn, None, type, season, plot_type, map_type)
                                    fig = vis.global_ensemble_plot([sim_seas_avgs, ref_seas_avgs], seas_avgs_diff, vn, season, type, vtres, title)

                        fig.savefig(plot_loc / plot_name, bbox_inches="tight")
                        plt.close(fig)
