#!/usr/bin/env python3
"""
AtmOcnGR.py

CVDP functions for plotting means, standard deviations, and trends.
License: MIT
"""

import numpy as np
from vis import *
from visualization.global_plots import *
from visualization.polar_plots import *
from utils import *

season_list = ["DJF", "JFM", "MAM", "JJA", "JAS", "SON", "ANN"]
var_seasons = {
    "psl": {"global": season_list + ["NDJFM"], "polar": season_list},
    "ts": season_list,
    "trefht": season_list,
    "prect": season_list,
}

nh_vars = ["NAM"]
sh_vars = ["SAM"]#, "PSA1", "PSA2"]
eof_vars = nh_vars + sh_vars

#ptypes = ["spatialmean"]
ptypes = ["spatialmean","trends"]
ptypes = ["trends"]
vns = ["psl"]
map_types = ["global"]
#map_types = ["global","polar"]

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
    
    ref_seas_avgs = kwargs["ref_seas_avgs"]
    sim_seas_avgs = kwargs["sim_seas_avgs"]
    seas_avgs_diff = kwargs["seas_avgs_diff"]

    ref_seas_trnds = kwargs["ref_seas_trnds"]
    sim_seas_trnds = kwargs["sim_seas_trnds"]
    seas_trnds_diff = kwargs["seas_trnds_diff"]

    print("AtmOcnGR ref_seas_trnds",ref_seas_trnds,"\n\n")
    unit = ref_seas_trnds.units

    res = get_variable_defaults()
    run = f"{sim_seas_avgs.run_name} - {ref_seas_avgs.run_name}"
    fig = None
    for vn in vns:
        for type in ptypes:
            for map_type in map_types:
                vres = res[vn]
                if map_type == "polar":
                    if vn == "psl":
                        vres = res["SAM"]
                vtres = vres[type]
                seasons = var_seasons[vn][map_type] if isinstance(var_seasons[vn], dict) else var_seasons[vn]

                for season in seasons:
                    for plot_type in ["summary", "indmem","indmemdiff"]:  # ,"indmemdiff"
                    #for plot_type in ["indmem"]:  # ,"indmemdiff"
                        #plot_name, title = get_plot_name_and_title(vn, '', type, season, plot_type, map_type)
                        if plot_type == "summary":
                            if map_type == "global":
                                if type == "spatialmean":
                                    plot_name, title = get_plot_name_and_title(vn, None, type, season, plot_type, map_type)
                                    fig = global_ensemble_plot([sim_seas_avgs, ref_seas_avgs], seas_avgs_diff, vn, season, type, vtres, title)
                                if type == "trends":
                                    if vn == "psl":
                                        if season == "NDJFM":
                                            var = "NPI"
                                            plot_name, title = get_plot_name_and_title(vn, None, type, season, plot_type, map_type)
                                            fig = global_ensemble_plot([sim_seas_trnds, ref_seas_trnds], seas_trnds_diff, vn, season, type, vtres, title)
                            if map_type == "polar":
                                if type == "trends":
                                    if vn == "psl":
                                        #for var in eof_vars:
                                        var = "SAM"
                                        plot_name, title = get_plot_name_and_title(vn, var, type, season, plot_type, map_type)
                                        fig = polar_ensemble_plot([sim_seas_trnds, ref_seas_trnds], seas_trnds_diff, vn, var, season, type, vtres, title)
                                #else:
                                    #polar_ensemble_plot(arrs, arr_diff, vn, var, season, ptype, plot_dict, title, debug=False)
                                #            fig = polar_ensemble_plot([sim_seas_avgs, ref_seas_avgs], seas_avgs_diff, vn, None, season, type, vtres, title)
                        if plot_type == "indmem":
                            if map_type == "global":
                                if type == "spatialmean":
                                    plot_name, title = get_plot_name_and_title(vn, None, type, season, plot_type, map_type)
                                    #global_indmem_latlon_plot(arrs, vn, season, plot_dict, title, ptype)
                                    fig = global_indmem_latlon_plot([sim_seas_avgs, ref_seas_avgs], vn, season, vtres, title, type)
                                if type == "trends":
                                    if vn == "psl":
                                        if season == "NDJFM":
                                            var = "NPI"
                                            plot_name, title = get_plot_name_and_title(vn, None, type, season, plot_type, map_type)
                                            #global_indmem_latlon_plot(arrs, vn, season, plot_dict, title, ptype)
                                            fig = global_indmem_latlon_plot([sim_seas_trnds, ref_seas_trnds], vn, season, vtres, title, type)
                            if map_type == "polar":
                                if type == "trends":
                                    if vn == "psl":
                                        var = "SAM"
                                        plot_name, title = get_plot_name_and_title(vn, var, type, season, plot_type, map_type)
                                        #polar_indmem_latlon_plot(vn, var, arrs, plot_dict, title, ptype)
                                        fig = polar_indmem_latlon_plot(vn, var, [sim_seas_trnds, ref_seas_trnds], vtres, title, plot_type,season)
                        if plot_type == "indmemdiff":
                            if map_type == "global":
                                if type == "spatialmean":
                                    plot_name, title = get_plot_name_and_title(vn, None, type, season, plot_type, map_type)
                                    fig = global_indmemdiff_latlon_plot(vn, run, unit, seas_avgs_diff, plot_type, vtres, title, season)
                                if type == "trends":
                                    if vn == "psl":
                                        if season == "NDJFM":
                                            var = "NPI"
                                            plot_name, title = get_plot_name_and_title(vn, None, type, season, plot_type, map_type)
                                            fig = global_indmemdiff_latlon_plot(vn, run, unit, seas_trnds_diff, plot_type, vtres, title, season)
                            if map_type == "polar":
                                if type == "trends":
                                    if vn == "psl":
                                        var = "SAM"
                                        #run = f"{sim_seas_avgs.run_name} - {ref_seas_avgs.run_name}"
                                        #polar_indmemdiff_latlon_plot(vn, var, run, arr, ptype, plot_dict, title)
                                        plot_name, title = get_plot_name_and_title(vn, var, type, season, plot_type, map_type)
                                        #polar_indmemdiff_latlon_plot(vn, var, run, arr, ptype, plot_dict, title)
                                        
                                        fig = polar_indmemdiff_latlon_plot(vn, var, run, unit, seas_trnds_diff, plot_type, vtres, title,season)

                        if not fig:
                            continue
                        else:
                            fig.savefig(plot_loc / plot_name, bbox_inches="tight")
                            plt.close(fig)
