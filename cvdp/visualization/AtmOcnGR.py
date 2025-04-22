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
        for type in ptypes:
            vres = res[vn]
            vtres = vres[type]

            seasons_ptypes = var_seasons[vn]
            seasons = seasons_ptypes[map_type]
            for season in seasons:
                for plot_type in ["summary","indmem"]: #,"indmemdiff"
                    if type == "spatialmean":
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
                    
                    if type == "trends":
                        print()

                    if plot_type == "summary":
                        fig = global_ensemble_plot([sim_seas_avgs,ref_seas_avgs], arr_diff, vn, season, type, vtres, title)
                        #fig.savefig(plot_loc / plot_name, bbox_inches="tight")
                        #plt.close(fig)

                    if plot_type == "indmem":
                        #global_indmem_latlon_plot(arrs, vn, season, plot_dict, title, ptype)
                        fig = global_indmem_latlon_plot([sim_seas_avgs,ref_seas_avgs], vn, season, vtres, title, type)
                        #fig.savefig(plot_loc / plot_name, bbox_inches="tight")
                        #plt.close(fig)

                    fig.savefig(plot_loc / plot_name, bbox_inches="tight")
                    plt.close(fig)
                    



# Plot functions
#---------------
def indmem_plot(finarrs, arrs, vn, var=None, season="ANN", ptype="trends", plot_dict=None, map_type="global", debug=False):

    if vn == "ts":
        plot_name = f"output/sst_{ptype}_{season.lower()}.indmem.png"
        title = f'SST {ptype.capitalize()} ({season.upper()})\n'
    elif season == "NDJFM":
        plot_name = f"output/npi_pattern_{season.lower()}.indmem.png"
        title = f'NPI Pattern ({season.upper()})\n'
    elif var in eof_vars:
        plot_name = f"output/{var.lower()}_pattern_{season.lower()}.indmem.png"
        title = f'{var} Pattern ({season.upper()})\n'
    else:
        plot_name = f"output/{vn}_{ptype}_{season.lower()}.indmem.png"
        title = f'{vn.upper()} {ptype.capitalize()} ({season.upper()})\n'

    if map_type == "global":
        global_indmem_latlon_plot(vn, finarrs, arrs, plot_dict, title, plot_name, ptype, season, debug)
    if map_type == "polar":
        stacked_polar_plot(vn, var, finarrs, arrs, plot_dict, title, plot_name, ptype, season, debug)


def indmemdiff_plot(finarrs, arr_diff, vn, var, season, ptype, plot_dict, map_type, debug=False):
    """
    d
    """

    #unit = finarrs[0].units
    try:
        run = f"{finarrs[0].run} - {finarrs[1].run}"
    except:
        print()
    try:
        run = f"{finarrs[0][vn].run} - {finarrs[1][vn].run}"
    except:
        print()
    run = "HAHAHAHJJPSå"
    print("vn:",vn,"\n")
    # Set file name and figure title
    if vn == "ts":
        plot_name = f"output/sst_{ptype}_{season.lower()}.indmemdiff.png"
        title = f'SST {ptype.capitalize()} Differences ({season.upper()})\n'
    elif season == "NDJFM":
        plot_name = f"output/npi_pattern_{season.lower()}.indmemdiff.png"
        title = f'NPI Pattern Differences ({season.upper()})\n'
    #elif var == "NAM" or var == "SAM":
    elif var in eof_vars:
        plot_name = f"output/{var.lower()}_pattern_{season.lower()}.indmemdiff.png"
        title = f'{var} Pattern Differences ({season.upper()})\n'
    else:
        print("no way its here?")
        plot_name = f"output/{vn}_{ptype}_{season.lower()}.indmemdiff.png"
        title = f'{vn.upper()} {ptype.capitalize()} Differences ({season.upper()})\n'

    if map_type == "global":
        print(title,"\n",plot_name)
        #global_diff_latlon_plot(vn, run, arr_diff, ptype, plot_dict, title, plot_name, debug)
    if map_type == "polar":
        print(title,"\n",plot_name)
        #polar_diff_plot(vn, var, run, arr_diff, ptype, plot_dict, title, plot_name, debug)



