#!/usr/bin/env python3
"""
AtmOcnGR.py

CVDP functions for plotting means, standard deviations, and trends.
License: MIT
"""

import numpy as np
from vis import *
from visualization.global_plots import *
from utils import *

season_list = ["DJF","JFM","MAM","JJA","JAS","SON","ANN"]
var_seasons = {"psl":{"global": season_list,"polar": season_list+["NDJFM"]},
               "ts": season_list,
               "trefht": season_list,
               "prect": season_list
               }

nh_vars = ["NAM"]
sh_vars = ["SAM", "PSA1", "PSA2"]
eof_vars = nh_vars+sh_vars
            
ptypes = ["spatialmean"]#,"trends","spatialstddev"
vns = ['psl']

map_type = "global"

full_dict = {}
title_deets = {}

#full_dict[]


def graphics(plot_loc, **kwargs):
    print("\nPlotting climatological seasonal means...")
    ref_seas_avgs = kwargs["ref_seas"]
    sim_seas_avgs = kwargs["sim_seas"]
    arr_diff = kwargs["diff_seas"]
    res = get_variable_defaults()
    for vn in vns:
        #ref_seas_avgs, sim_seas_avgs, arr_diff = mean_seasonal_calc(ref_datasets[ref_0][vn], sim_datasets[sim_0][vn])
        for type in ptypes:
            vres = res[vn]
            vtres = vres[type]

            seasons_ptypes = var_seasons[vn]
            seasons = seasons_ptypes[map_type]
            for season in seasons:
                for plot_type in ["summary","indmem"]: #,"indmemdiff"
                    if map_type == "global":
                        if vn == "ts":
                            plot_name = f"sst_{type}_{season.lower()}.{plot_type}.png"
                            if plot_type == "summary":
                                title = f'Ensemble Summary: SST {type.capitalize()} ({season.upper()})'
                            if plot_type == "indmem":
                                title = f'SST {type.capitalize()} ({season.upper()})\n'
                            if plot_type == "indmemdiff":
                                title = f'SST {type.capitalize()} Differences ({season.upper()})\n'
                        else:
                            plot_name = f"{vn}_{type}_{season.lower()}.{plot_type}.png"
                            if plot_type == "summary":
                                title = f'Ensemble Summary: {vn.upper()} {type.capitalize()} ({season.upper()})'
                            if plot_type == "indmem":
                                title = f'{vn.upper()} {type.capitalize()} ({season.upper()})\n'
                            if plot_type == "indmemdiff":
                                title = f'{vn.upper()} {type.capitalize()} Differences ({season.upper()})\n'

                    if map_type == "polar":
                        if vn == "ts":
                            plot_name = f"sst_{type}_{season.lower()}.{plot_type}.png"
                            if plot_type == "summary":
                                title = f'Ensemble Summary: SST {type.capitalize()} ({season.upper()})'
                        elif season == "NDJFM":
                            plot_name = f"npi_pattern_{season.lower()}.{plot_type}.png"
                            if plot_type == "summary":
                                title = f'Ensemble Summary: NPI Pattern ({season.upper()})'
                            if plot_type == "indmemdiff":
                                title = f'NPI Pattern Differences ({season.upper()})\n'
                            if plot_type == "indmem":
                                title = f'NPI Pattern ({season.upper()})\n'
                        elif vn in eof_vars:
                            print("EOF func")
                            plot_name = f"{vn.lower()}_pattern_{season.lower()}.{plot_type}.png"
                            if plot_type == "summary":
                                title = f'Ensemble Summary: {vn} Pattern ({season.upper()})\n'
                            if plot_type == "indmem":
                                title = f'{vn} Pattern ({season.upper()})\n'
                            if plot_type == "indmemdiff":
                                title = f'{vn} Pattern Differences ({season.upper()})\n'
                        else:
                            plot_name = f"{vn}_{type}_{season.lower()}.{plot_type}.png"
                            if plot_type == "summary":
                                title = f'Ensemble Summary: {vn.upper()} {type.capitalize()} ({season.upper()})'

                    if plot_type == "summary":
                        fig = global_ensemble_plot([sim_seas_avgs,ref_seas_avgs], arr_diff, vn, season, type, vtres, title)

                    if plot_type == "indmem":
                        fig = global_indmem_latlon_plot([sim_seas_avgs,ref_seas_avgs], vn, season, vtres, title, type)

                    fig.savefig(plot_loc / plot_name, bbox_inches="tight")
                    plt.close(fig)
