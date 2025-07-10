#!/usr/bin/env python3
"""
AtmOcnGR.py

CVDP functions for plotting means, standard deviations, trends, timeseries, metrics tables, etc.
License: MIT
"""

import numpy as np
from vis.global_plots import *
#from cvdp.vis.global_plots import *
import matplotlib.pyplot as plt
import cvdp_utils.utils as helper_utils
import cvdp_utils.analysis as an
#import cvdp.cvdp_utils.utils as helper_utils

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

ptypes = ["spatialmean", "trends"]
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
    print("\n\nseas_avgs_diff",seas_avgs_diff,"\n\n")
    res = helper_utils.get_variable_defaults()
    vn = kwargs["vn"]
    #for vn in vns:
    if 1==1:
        #unit = ref_seas_avgs.units
        for type in ptypes:
            fig = None
            for map_type in map_types:
                vres = res[vn]
                vtres = vres[type]
                seasons = var_seasons[vn][map_type] if isinstance(var_seasons[vn], dict) else var_seasons[vn]

                for season in seasons:
                    for plot_type in ["summary", "indmem","indmemdiff"]:
                        if plot_type == "summary":
                            if map_type == "global":
                                """if type == "spatialmean":
                                    sim_seas_avg = sim_seas_avgs[f"{vn}_{type}_{season.lower()}"].mean(dim="time")
                                    ref_seas_avg = ref_seas_avgs[f"{vn}_{type}_{season.lower()}"].mean(dim="time")
                                    seas_avg_diff = seas_avgs_diff[f"{vn}_{type}_{season.lower()}"].mean(dim="time")
                                    plot_name, title = get_plot_name_and_title(vn, None, type, season, plot_type, map_type)
                                    fig = global_ensemble_plot([sim_seas_avg, ref_seas_avg], seas_avg_diff, vn, type, vtres, title)"""
                                if type == "trends":
                                    sim_seas_avg, sim_res, sim_fit = af.lin_regress(sim_seas_avgs[f"{vn}_{type}_{season.lower()}"])
                                    ref_seas_avg, ref_res, res_fit = af.lin_regress(ref_seas_avgs[f"{vn}_{type}_{season.lower()}"])
                                    # If the cases are different shapes, we need to interpolate one to the other first
                                    #NOTE: the value that comes out of interp_diff is either None, or interpolated difference array
                                    arr_prime = an.interp_diff(sim_seas_avg, ref_seas_avg)

                                    # If arr_prime is None, then the two runs have already been interpolated (TS -> SST) or are the same grid/shape
                                    if arr_prime is None:
                                        seas_avg_diff = sim_seas_avg - ref_seas_avg
                                    else:
                                        seas_avg_diff = (arr_prime - ref_seas_avg)

                                    if vn == "psl":
                                        if season == "NDJFM":
                                            var = "NPI"
                                        else:
                                            var = vn
                                        plot_name, title = get_plot_name_and_title(vn, var, type, season, plot_type, map_type)
                                        fig = global_ensemble_plot([sim_seas_avg, ref_seas_avg], seas_avg_diff, vn, type, vtres, title)
                        """if plot_type == "indmem":
                            if map_type == "global":
                                if type == "spatialmean":
                                    sim_seas_avg = sim_seas_avgs[f"{vn}_{type}_{season.lower()}"].mean(dim="time")
                                    ref_seas_avg = ref_seas_avgs[f"{vn}_{type}_{season.lower()}"].mean(dim="time")
                                    seas_avg_diff = seas_avgs_diff[f"{vn}_{type}_{season.lower()}"].mean(dim="time")
                                    plot_name, title = get_plot_name_and_title(vn, None, type, season, plot_type, map_type)
                                    #global_indmem_latlon_plot(arrs, vn, season, plot_dict, title, ptype)
                                    #fig = global_indmem_latlon_plot([sim_seas_avgs, ref_seas_avgs], vn, season, vtres, title, type)
                                    fig = global_indmem_latlon_plot(vn, [sim_seas_avg, ref_seas_avg], vtres, title, plot_type)
                        if plot_type == "indmemdiff":
                            if map_type == "global":
                                if type == "spatialmean":
                                    seas_avg_diff = seas_avgs_diff[f"{vn}_{type}_{season.lower()}"].mean(dim="time")
                                    plot_name, title = get_plot_name_and_title(vn, None, type, season, plot_type, map_type)
                                    #global_indmemdiff_latlon_plot(vn, run, arr, ptype, plot_dict, title)
                                    run = f"{sim_seas_avg.run.values} - {ref_seas_avg.run.values}"
                                    fig = global_indmemdiff_latlon_plot(vn, run, seas_avg_diff, plot_type, vtres, title)"""
                        if fig is not None:
                            fig.savefig(plot_loc / plot_name, bbox_inches="tight")
                            plt.close(fig)
