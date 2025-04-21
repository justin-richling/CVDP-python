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
var_seasons = {"psl": season_list+["NDJFM"],
               "ts": season_list,
               "trefht": season_list,
               "prect": season_list
               }

nh_vars = ["NAM"]
sh_vars = ["SAM", "PSA1", "PSA2"]
eof_vars = nh_vars+sh_vars
            
ptypes = ["spatialmean"]#,"trends","spatialstddev"

map_type = "global"

full_dict = {}
title_deets = {}

#full_dict[]


def graphics(plot_loc, **kwargs):
    print("\nPlotting climatological seasonal means...")
    ref_seas_avgs = kwargs["ref_seas"]
    sim_seas_avgs = kwargs["sim_seas"]
    arr_diff = kwargs["diff_seas"]
    vns = ['psl']
    res = get_variable_defaults()
    for vn in vns:
        for type in ptypes:
            vres = res[vn]
            vtres = vres[type]

            seasons = var_seasons[vn]
            for season in seasons:
                for plot_type in ["summary","indmem","indmemdiff"]:
                    if type == "spatialmean":
                        if map_type == "global":
                            if vn == "ts":
                                plot_name = f"sst_{type}_{season.lower()}.{plot_type}.png"
                                if plot_type == "summary":
                                    title = f'Ensemble Summary: SST {type.capitalize()} ({season.upper()})'
                            else:
                                plot_name = f"{vn}_{type}_{season.lower()}.{plot_type}.png"
                                if plot_type == "summary":
                                    title = f'Ensemble Summary: {vn.upper()} {type.capitalize()} ({season.upper()})'
                        
                        fig = global_ensemble_plot([sim_seas_avgs,ref_seas_avgs], arr_diff, vn, season, type, vtres, title)
                        #global_sim_ref_plot(vn, finarrs, arrs, plot_dict, title, plot_name, ptype, season, debug)

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

                    fig.savefig(plot_loc / plot_name, bbox_inches="tight")



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
        global_sim_ref_plot(vn, finarrs, arrs, plot_dict, title, plot_name, ptype, season, debug)
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
        #global_latlon_diff_plot(vn, run, arr_diff, ptype, plot_dict, title, plot_name, debug)
    if map_type == "polar":
        print(title,"\n",plot_name)
        #polar_diff_plot(vn, var, run, arr_diff, ptype, plot_dict, title, plot_name, debug)






'''def ensemble_plot(arrs, arr_diff, vn, var, season, ptype, plot_dict, map_type):
    """
    arrs
    arr_diff
    vn
    var=None
    season="ANN"
    ptype="trends"
    plot_dict=None
    map_type="global"

    """
    #if not var:
    #    var = vn

    print("var",var,"\n")
    # Set file name and figure title
    if vn == "ts":
        plot_name = f"output/sst_{ptype}_{season.lower()}.summary.png"
        #title = f'SST {ptype.capitalize()} ({season})\n'
        title = f'Ensemble Summary: SST {ptype.capitalize()} ({season.upper()})'
    elif season == "NDJFM":
        plot_name = f"output/npi_pattern_{season.lower()}.summary.png"
        #npi_pattern_ndjfm.summary.png
        #title = f'NPI Pattern ({season})\n'
        title = f'Ensemble Summary: NPI Pattern ({season.upper()})'
    elif var in eof_vars:
        print("EOF func")
        plot_name = f"output/{var.lower()}_pattern_{season.lower()}.summary.png"
        title = f'Ensemble Summary: {var} Pattern ({season.upper()})\n'
    else:
        print("OR HERE?")
        plot_name = f"output/{vn}_{ptype}_{season.lower()}.summary.png"
        #title = f'{vn.upper()} {ptype.capitalize()} ({season})\n'
        title = f'Ensemble Summary: {vn.upper()} {ptype.capitalize()} ({season.upper()})'

    if map_type == "global":
        print(map_type)
        fig = global_enesmble_plot(arrs, arr_diff, vn, season, ptype, plot_dict, title)
    if map_type == "polar":
        print(map_type)
        polar_ensemble_plot(finarrs, arrs, arr_diff, vn, var, season, ptype, plot_dict, title, plot_name)
    return fig'''

#vns = ["ts", "psl"]
vns = ["psl"]
season = "SON"

"""
#finarrs = [ts_cesm_avg,ts_obs_avg]
for i,vn in enumerate(vns):
    print("vn",vn,"\n---------------------\n")
    debug = True
    finarrs = finarrs_dict[vn] # = [ts_cesm_avg,ts_obs_avg]
    #if vn == "ts":
    #   debug = True
    for ptype in ptypes:
        print("ptype",ptype,"\n---------------------\n")

        #arr_var = ts_cesm_avg[f"{vn}_{ptype}_{season.lower()}"]
        #arr_var2 = ts_obs_avg[f"{vn}_{ptype}_{season.lower()}"]

        arr_var = finarrs[0][f"{vn}_{ptype}_{season.lower()}"]
        arr_var2 = finarrs[1][f"{vn}_{ptype}_{season.lower()}"]

        arrs_raw = [arr_var,arr_var2]

        arrs = []
        for i in arrs_raw:
            if vn == "ts":
                # interp to mask
                i = an.interp_mask(i, lsmask)
            if ptype == "trends":
                arr, res, fit = af.lin_regress(i)
            else:
                arr = i.mean(dim="time")
            arrs.append(arr)

        # Attempt to get difference data
        #-------------------------------
        arr_anom1 = arrs[0]
        arr_anom2 = arrs[1]

        # If the cases are different shapes, we need to interpolate one to the other first
        #NOTE: the value that comes out of interp_diff is either None, or interpolated difference array
        arr_prime = an.interp_diff(arr_anom1, arr_anom2)

        #print("arr_prime type:",type(arr_prime),"\n")
        # If arr_prime is None, then the two runs have already been interpolated (TS -> SST) or are the same grid/shape
        if arr_prime is None:
            arr_diff = arr_anom1 - arr_anom2
        else:
            arr_diff = (arr_prime - arr_anom2)

        # Plot details dict
        pdict = plot_dict[ptype]

        # Stacked lat/lon plot of the two runs
        indmem_plot(finarrs, arrs, vn, None, season, ptype, pdict, map_type="global", debug=debug)

        # Single plot of differences
        indmemdiff_plot(finarrs, arr_diff, vn, None, season, ptype, pdict, map_type="global", debug=debug)

        # Four panel plot of run, obs, differences, and rank
        ensemble_plot(finarrs, arrs, arr_diff, vn, None, season, ptype, pdict, map_type="global", debug=debug)
"""



