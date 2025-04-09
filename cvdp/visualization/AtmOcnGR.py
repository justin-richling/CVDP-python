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

def AtmOcnGR(plot_loc, **kwargs):
    ref_seas_avgs = kwargs["ref_seas"]
    sim_seas_avgs = kwargs["sim_seas"]
    arr_diff = kwargs["diff_seas"]
    vn = 'psl'
    res = get_variable_defaults()
    tres = res[type]
    vtres = tres[vn]
    for type in ptypes:

        #ensemble_plot(arrs, arr_diff, vn, var=None, season="ANN", ptype="trends", plot_dict=None, map_type="global", debug=False)
        season = "SON"
        global_ensemble_fig = ensemble_plot([sim_seas_avgs,ref_seas_avgs], arr_diff, vn, "PSL", season, type, vtres, "global")
        global_ensemble_fig.savefig(plot_loc / f"psl_ensemble_{season.lower()}.png",bbox_inches="tight")


    """plot_dict_mean = {"psl": {"range": np.linspace(968,1048,21),
                            "ticks": np.arange(976,1041,8),
                            #"cbarticks":"",
                            #"diff_cbarticks":np.arange(-10,11,2),
                            "diff_range": np.arange(-11,12,1),
                            "diff_ticks": np.arange(-10,11,1),
                            #"cmap": cm.get_NCL_colormap("amwg256", extend='None'),#amwg_cmap,
                            "cmap": amwg_cmap,
                            "units":"hPa"},
                    "ts": {"range": np.linspace(-2,38,21),
                            "ticks": np.linspace(-2,38,21),
                            #"ticks": np.arange(0,38,2),
                            #"tick_labels": np.arange(0,38,2),
                            "cbarticks": np.arange(0,37,2),
                            "diff_cbarticks":np.arange(-5,6,1),
                            "diff_range": np.arange(-5.5,5.6,0.5),
                            "diff_ticks": np.arange(-5.5,5.6,0.5),
                            #"diff_ticks": np.arange(-5,6,1),
                            "cmap": amwg_cmap,
                            "units":"C"}
                }

    plot_dict_trends = {"psl": {"range": np.linspace(-9,9,19),
                                "ticks": np.arange(-8, 9, 1),
                                "cbarticks": np.arange(0,37,2),
                                "diff_cbarticks":np.arange(-8, 9, 1),
                                "cmap": amwg_cmap,
                                "units":"hPa"},
                        "ts": {"range": [-8, -7, -6, -5, -4, -3, -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8],
                                #"ticks": [-6, -4, -2, -0.5, 0, 0.5, 2, 4, 6],
                                "ticks": [-8, -7, -6, -5, -4, -3, -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8],
                                #"diff_range": np.arange(-5,6,1),
                                #"diff_ticks": np.arange(-5,6,1),
                                "cbarticks": [-6, -4, -2, -0.5, 0, 0.5, 2, 4, 6],
                                "cmap": amwg_cmap,
                                "units":"C"},
                        "NAM": {"range": np.linspace(-8, 8, 17),
                                "ticks": np.arange(-7,8,1),
                                "cmap": amwg_cmap,
                                "units": "hPa"},
                        "PNA": {"range": np.linspace(-8, 8, 17),
                                "ticks": np.arange(-7,8,1),
                                "cmap": amwg_cmap,
                                "units": "hPa"},
                        "PNO": {"range": np.linspace(-8, 8, 17),
                                "ticks": np.arange(-7,8,1),
                                "cmap": amwg_cmap,
                                "units": "hPa"},
                        "SAM": {"range": np.linspace(-8, 8, 17),
                                "ticks": np.arange(-7,8,1),
                                "cmap": amwg_cmap,
                                "units": "hPa"},
                        "PSA1": {"range": np.linspace(-8, 8, 17),
                                "ticks": np.arange(-7,8,1),
                                "cmap": amwg_cmap,
                                "units": "hPa"},
                        "PSA2": {"range": np.linspace(-8, 8, 17),
                                "ticks": np.arange(-7,8,1),
                                "cmap": amwg_cmap,
                                "units": "hPa"}
                }"""


    #plot_dict = {"spatialmean": plot_dict_mean,
    #            "trends": plot_dict_trends}


# Plot functions
#---------------
def indmem_plot(finarrs, arrs, vn, var=None, season="ANN", ptype="trends", plot_dict=None, map_type="global", debug=False):

    if vn == "ts":
        plot_name = f"output/sst_{ptype}_{season.lower()}.indmem.png"
        title = f'SST {ptype.capitalize()} ({season.upper()})\n'
    elif season == "NDJFM":
        plot_name = f"output/npi_pattern_{season.lower()}.indmem.png"
        title = f'NPI Pattern ({season.upper()})\n'
    #elif var == "NAM" or var == "SAM":
    elif var in eof_vars:
        plot_name = f"output/{var.lower()}_pattern_{season.lower()}.indmem.png"
        title = f'{var} Pattern ({season.upper()})\n'
    else:
        plot_name = f"output/{vn}_{ptype}_{season.lower()}.indmem.png"
        title = f'{vn.upper()} {ptype.capitalize()} ({season.upper()})\n'

    if map_type == "global":
        stacked_global_latlon_plot(vn, finarrs, arrs, plot_dict, title, plot_name, ptype, season, debug)
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






def ensemble_plot(arrs, arr_diff, vn, var=None, season="ANN", ptype="trends", plot_dict=None, map_type="global", debug=False):
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
        print("IS IT HERE?")
        plot_name = f"output/{var.lower()}_pattern_{season.lower()}.summary.png"
        title = f'Ensemble Summary: {var} Pattern ({season.upper()})\n'
    else:
        print("OR HERE?")
        plot_name = f"output/{vn}_{ptype}_{season.lower()}.summary.png"
        #title = f'{vn.upper()} {ptype.capitalize()} ({season})\n'
        title = f'Ensemble Summary: {vn.upper()} {ptype.capitalize()} ({season.upper()})'

    if map_type == "global":
        print(map_type)
        #global_ensemble_plot(arrs, arr_diff, vn, season, ptype, plot_dict, title, debug=False)
        ####global_ensemble_plot(finarrs, arrs, arr_diff, vn, season, ptype, plot_dict, title, plot_name, debug)
        fig = global_ensemble_plot(arrs, arr_diff, vn, season, ptype, plot_dict, title, debug)
    if map_type == "polar":
        print(map_type)
        polar_ensemble_plot(finarrs, arrs, arr_diff, vn, var, season, ptype, plot_dict, title, plot_name, debug)
    return fig


#vn = "ts"
#vns = ["ts", "psl"]
vns = ["psl"]
#vns = ["ts"]
#ptypez = ["spatialmean"]
season = "SON"

#finarrs_dict
#ptypes
#lsmask
#plot_dict
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



