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

from vis.polar_plots import (
    polar_ensemble_plot,
    polar_indmem_latlon_plot,
    polar_indmemdiff_latlon_plot
)
import cvdp_utils.avg_functions as af
import cvdp_utils.utils as helper_utils
import cvdp_utils.analysis as an
#import cvdp.cvdp_utils.utils as helper_utils

season_list = ["DJF", "JFM", "MAM", "JJA", "JAS", "SON", "ANN"]
season_list = ["DJF"]
var_seasons = {
    "psl": {"global": season_list + ["NDJFM"], "polar": season_list},
    "ts": season_list,
    "trefht": season_list,
    "prect": season_list,
}

# EOF variables
nh_vars = ["NAM"]
sh_vars = ["SAM", "PSA1", "PSA2"]
eof_vars = nh_vars + sh_vars

eof_vars = ["NAM", "SAM", "PSA1", "PSA2"]

ptypes = ["spatialmean", "trends"]
vns = ["psl"]
map_types = ["global", "polar"]
plot_types = ["summary", "indmem", "indmemdiff"]
#plot_types = ["indmemdiff"]


'''def get_plot_name_and_title(vn, var, ptype, season, plot_type, map_type):
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
        if vn == "psl":
            for eof_var in eof_vars:
                plot_name = f"{eof_var.lower()}_pattern_{season_lower}.{plot_type}.png"
                title = {
                    "summary": f"Ensemble Summary: {eof_var} Pattern ({season_upper})\n",
                    "indmem": f"{eof_var} Pattern ({season_upper})\n",
                    "indmemdiff": f"{eof_var} Pattern Differences ({season_upper})\n",
                }
            """plot_name = f"{var.lower()}_pattern_{season_lower}.{plot_type}.png"
            title_map = {
                "summary": f"Ensemble Summary: {var} Pattern ({season_upper})\n",
                "indmem": f"{var} Pattern ({season_upper})\n",
                "indmemdiff": f"{var} Pattern Differences ({season_upper})\n",
            }"""
        else:
            plot_name = f"{base_var}_pattern_{season_lower}.{plot_type}.png"
    else:
        plot_name = "unknown.png"

    title = {
        "summary": f"Ensemble Summary: {var or title_var} {ptype.capitalize()} ({season_upper})",
        "indmem": f"{var or title_var} {ptype.capitalize()} ({season_upper})\n",
        "indmemdiff": f"{var or title_var} {ptype.capitalize()} Differences ({season_upper})\n",
    }.get(plot_type, "Unknown Title")

    return plot_name, title'''


def get_plot_name_and_title(vn, var, ptype, season, plot_type, map_type):
    """
    Returns one or more (plot_name, title) tuples depending on vn/map_type.
    """
    season_upper = season.upper()
    season_lower = season.lower()

    base_var = "sst" if vn == "ts" else vn
    title_var = "SST" if vn == "ts" else vn.upper()

    results = []

    if map_type == "global":
        if ptype == "trends" and vn == "psl" and season == "NDJFM":
            plot_name = f"npi_pattern_{season_lower}.{plot_type}.png"
            title = {
                "summary": f"Ensemble Summary: NPI Pattern ({season_upper})",
                "indmem": f"NPI Pattern ({season_upper})\n",
                "indmemdiff": f"NPI Pattern Differences ({season_upper})\n",
            }[plot_type]
            results.append((plot_name, title))
        else:
            suffix = f"{ptype}_{season_lower}.{plot_type}.png"
            plot_name = f"{base_var}_{suffix}" if ptype != "trends" else f"{base_var}_pattern_{season_lower}.{plot_type}.png"
            title = {
                "summary": f"Ensemble Summary: {title_var} {ptype.capitalize()} ({season_upper})",
                "indmem": f"{title_var} {ptype.capitalize()} ({season_upper})\n",
                "indmemdiff": f"{title_var} {ptype.capitalize()} Differences ({season_upper})\n",
            }[plot_type]
            results.append((plot_name, title))

    elif map_type == "polar" and vn == "psl":
        # Return one entry per EOF mode
        eof_vars = ["NAM", "SAM", "PSA1", "PSA2"]
        #for eof_var in eof_vars:
        if var in eof_vars:
            plot_name = f"{var.lower()}_pattern_{season_lower}.{plot_type}.png"
            title = {
                "summary": f"Ensemble Summary: {var} Pattern ({season_upper})",
                "indmem": f"{var} Pattern ({season_upper})\n",
                "indmemdiff": f"{var} Pattern Differences ({season_upper})\n",
            }[plot_type]
            results.append((plot_name, title))
    else:
        plot_name = f"{base_var}_pattern_{season_lower}.{plot_type}.png"
        title = {
            "summary": f"Ensemble Summary: {title_var} {ptype.capitalize()} ({season_upper})",
            "indmem": f"{title_var} {ptype.capitalize()} ({season_upper})\n",
            "indmemdiff": f"{title_var} {ptype.capitalize()} Differences ({season_upper})\n",
        }[plot_type]
        results.append((plot_name, title))

    return results




def compute_mean_diff(sim, ref):
    arr_prime = an.interp_diff(sim, ref)
    return sim - ref if arr_prime is None else arr_prime - ref


def compute_trend(data):
    return af.lin_regress(data)[0]  # returns the trend array


def handle_plot(plot_type, ptype, map_type, vn, season, vtres, sim_data, ref_data, var=None):
    sim = sim_data.mean(dim="time") if ptype == "spatialmean" else af.lin_regress(sim_data)[0]
    ref = ref_data.mean(dim="time") if ptype == "spatialmean" else af.lin_regress(ref_data)[0]

    arr_prime = an.interp_diff(sim, ref)
    #diff = arr_prime if arr_prime is not None else (sim - ref)

    # If arr_prime is None, then the two runs have already been interpolated (TS -> SST) or are the same grid/shape
    if arr_prime is None:
        diff = sim - ref
    else:
        diff = (arr_prime - ref)

    #print("DIFF PLOT BOI",diff,"\n\n")

    results = []
    #fig = None
    print("\nHANDLE PLOTS:",vn, var, ptype, season, plot_type, map_type,"\n")
    for plot_name, title in get_plot_name_and_title(vn, var, ptype, season, plot_type, map_type):
        print("\tplot_name",plot_name)
        if plot_type == "summary":
            if map_type == "polar":
                #polar_ensemble_plot(arrs, arr_diff, vn, var, ptype, plot_dict, title, debug=False)
                polar_ensemble_plot([sim, ref], diff, vn, var, ptype, vtres, title)
            if map_type == "global":
                fig = global_ensemble_plot([sim, ref], diff, vn, ptype, vtres, title)
        elif plot_type == "indmem":
            if map_type == "polar":
                #polar_indmem_latlon_plot(vn, var, arrs, plot_dict, title, ptype)
                fig = polar_indmem_latlon_plot(vn, var, [sim, ref], vtres, title, ptype)
            if map_type == "global":
                fig = global_indmem_latlon_plot(vn, [sim, ref], vtres, title, plot_type)
        elif plot_type == "indmemdiff":
        #if plot_type == "indmemdiff":
            run = f"{sim.run.values} - {ref.run.values}"
            if map_type == "polar":
                # polar_indmemdiff_latlon_plot(vn, var, run, unit, arr, ptype, plot_dict, title)
                fig = polar_indmemdiff_latlon_plot(vn, var, run, diff, ptype, vtres, title)
            if map_type == "global":
                fig = global_indmemdiff_latlon_plot(vn, run, diff, plot_type, vtres, title)
        else:
            fig = None
        if fig:
            results.append((fig, plot_name))
    return results




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

                    # Use EOF vars for polar PSL
                    if vn == "psl" and map_type == "polar":
                        for var in eof_vars:
                            results = handle_plot(plot_type, ptype, map_type, vn, season, vtres, sim_data, ref_data, var=var)

                            for fig, plot_name in results:
                                fig.savefig(plot_loc / plot_name, bbox_inches="tight")
                                plt.close(fig)
                    else:
                        results = handle_plot(plot_type, ptype, map_type, vn, season, vtres, sim_data, ref_data)

                        for fig, plot_name in results:
                            fig.savefig(plot_loc / plot_name, bbox_inches="tight")
                            plt.close(fig)
                """
                if type == "trends":
                                    if vn == "psl":
                                        if season == "NDJFM":
                                            var = "NPI"
                                        else:
                                            var = vn
                                        plot_name, title = get_plot_name_and_title(vn, var, type, season, plot_type, map_type)
                """
                
                """else:
                    results = handle_plot(plot_type, ptype, map_type, vn, season, vtres, sim_data, ref_data)

                    for fig, plot_name in results:
                        fig.savefig(plot_loc / plot_name, bbox_inches="tight")
                        plt.close(fig)"""
