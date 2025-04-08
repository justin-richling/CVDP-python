#!/usr/bin/env python
"""
cli.py

Command Line Interface (CLI) for CVDP.

Parses user input from command line and passes arguments to automation in cvdp.py
"""
import xarray as xr
import argparse
from importlib.metadata import version as getVersion
#import cvdp
"""from cvdp.scripts.namelist import createNameList
from cvdp.scripts.atm_ocn_mean_stddev_calc import calcAtmOcnMeanStd
from cvdp.scripts.atm_mean_stddev_gr import calcAtmOcnMeanStdGR
"""
#from cvdp.visualization.AtmOcnGR import *
from visualization.AtmOcnGR import *
from definitions import * #PARENT_DIR,PATH_VARIABLE_DEFAULTS
from vis import *

def main():
    #parser = argparse.ArgumentParser(description = f"Command Line Interface (CLI) for Climate Variability and Diagnostics Package (CVDP) Version {getVersion('cvdp')}")
    parser = argparse.ArgumentParser(description = f"Command Line Interface (CLI) for Climate Variability and Diagnostics Package (CVDP) Version 0.0.1")
    parser.add_argument("output_dir", nargs = 1, metavar = "output_dir", type = str, help = "Path to output directory.")
    #parser.add_argument("ref_yml", nargs = 1, metavar = "ref_yml", type = str, help = "Path to reference dataset YML file.")
    #parser.add_argument("sim_yml", nargs = 1, metavar = "sim_yml", type = str, help = "Path to simulation dataset YML file.")
    parser.add_argument("-c", nargs = 1, metavar = "--config", type = str, help = "Optional path to YML file to override default variable configurations.")

    args = parser.parse_args()
    var_configs = args.c

    from pathlib import Path
    plot_loc = Path(args.output_dir[0])
    if not plot_loc.is_dir():
        print(f"\tINFO: Directory not found, making new plot save location")
        plot_loc.mkdir(parents=True)
    
    if args.c is None:
        var_configs = PATH_VARIABLE_DEFAULTS
        #var_configs = cvdp.definitions.PATH_VARIABLE_DEFAULTS
    else:
        var_configs = args.c[0]

    from pathlib import Path
    def check_or_save_nc(save_loc, clobber, var_data_array=None):
        if Path(save_loc).is_file() and not clobber:
            var_data_array = xr.open_mfdataset(save_loc,coords="minimal", compat="override", decode_times=True)
            return var_data_array
        else:
            #var_data_array = read_datasets(paths, ds_info["variable"], [syr, eyr], mems)
            #Path(save_loc).unlink(missing_ok=True)
            #var_data_array.to_netcdf(save_loc)
            return None

    from file_io import get_input_data
    #from io import get_input_data
    #from cvdp.io import get_input_data
    from diag import compute_seasonal_avgs, compute_seasonal_stds
    #from cvdp.diag import compute_seasonal_avgs, compute_seasonal_stds
    from vis import plot_seasonal_means
    #from cvdp.vis import plot_seasonal_ensemble_means, CVDPNotebook

    from definitions import PARENT_DIR
    #from cvdp.definitions import parent_dir
    #ref_datasets, sim_datasets = get_input_data("../example_config.yaml")
    ref_datasets, sim_datasets = get_input_data(f"{PARENT_DIR}/example_config.yaml")

    vn = "psl"
    ref_0 = list(ref_datasets.keys())[0]
    sim_0 = list(sim_datasets.keys())[0]
    ref_seas_avgs = compute_seasonal_avgs(ref_datasets[ref_0][vn])
    sim_seas_avgs = compute_seasonal_avgs(sim_datasets[sim_0][vn])
    print("AHHHH",sim_seas_avgs,"\n\n")

    seasonal_ensemble_fig = plot_seasonal_means(sim_seas_avgs)
    seasonal_ensemble_fig.savefig(plot_loc / "my_plot.png")

    import old_utils.analysis as an

    #arr_var = finarrs[0][f"{vn}_{ptype}_{season.lower()}"]
    #arr_var2 = finarrs[1][f"{vn}_{ptype}_{season.lower()}"]

    ptype = "spatialmean"

    """arrs_raw = [ref_seas_avgs,sim_seas_avgs]

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
    arr_anom2 = arrs[1]"""

    # If the cases are different shapes, we need to interpolate one to the other first
    #NOTE: the value that comes out of interp_diff is either None, or interpolated difference array
    #arr_prime = an.interp_diff(arr_anom1, arr_anom2)
    arr_prime = an.interp_diff(ref_seas_avgs, sim_seas_avgs)


    #print("arr_prime type:",type(arr_prime),"\n")
    # If arr_prime is None, then the two runs have already been interpolated (TS -> SST) or are the same grid/shape
    if arr_prime is None:
        #arr_diff = arr_anom1 - arr_anom2
        arr_diff = ref_seas_avgs - sim_seas_avgs
    else:
        #arr_diff = (arr_prime - arr_anom2)
        arr_diff = (arr_prime - sim_seas_avgs)



    """# Timeseries Plot
    #----------------
    #timeseries_plot(var, season, test, obs)
    time_series_fig = timeseries_plot(var=vn, test_da=sim_seas_avgs, obs_da=ref_seas_avgs)
    time_series_fig(plot_loc / "psl_timeseries_djf.png",bbox_inches="tight")"""

    plot_dict_mean = {"psl": {"range": np.linspace(968,1048,21),
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
                }


    plot_dict = {"spatialmean": plot_dict_mean,
                "trends": plot_dict_trends}
    
    #ensemble_plot(arrs, arr_diff, vn, var=None, season="ANN", ptype="trends", plot_dict=None, map_type="global", debug=False)
    global_ensemble_fig = ensemble_plot([sim_seas_avgs,ref_seas_avgs], arr_diff, vn, "PSL","DJF", ptype, plot_dict[ptype], "global")
    global_ensemble_fig.savefig(plot_loc / "psl_ensemble_djf.png",bbox_inches="tight")

#ensemble_avgs = seasonal_avgs.mean(dim="member").compute()
if __name__ == '__main__':
    main()