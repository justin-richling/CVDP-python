#!/usr/bin/env python
"""
cli.py

Command Line Interface (CLI) for CVDP.

Parses user input from command line and passes arguments to automation in cvdp.py
"""

import argparse
from importlib.metadata import version as getVersion

#import cvdp
"""from cvdp.scripts.namelist import createNameList
from cvdp.scripts.atm_ocn_mean_stddev_calc import calcAtmOcnMeanStd
from cvdp.scripts.atm_mean_stddev_gr import calcAtmOcnMeanStdGR
"""

#from diag.AtmOcnMean import *
from diag.AtmOcnMean import mean_seasonal_calc
#from vis.AtmOcnGR import *
from vis.AtmOcnGR import graphics
from definitions import * #PARENT_DIR,PATH_VARIABLE_DEFAULTS

#from cvdp.diag.AtmOcnMean import *
#from cvdp.vis.AtmOcnGR import *
#from cvdp.definitions import * #PARENT_DIR,PATH_VARIABLE_DEFAULTS
#from vis import *

def main():
    #parser = argparse.ArgumentParser(description = f"Command Line Interface (CLI) for Climate Variability and Diagnostics Package (CVDP) Version {getVersion('cvdp')}")
    parser = argparse.ArgumentParser(description = f"Command Line Interface (CLI) for Climate Variability and Diagnostics Package (CVDP) Version 0.0.1")
    parser.add_argument("output_dir", nargs = 1, metavar = "output_dir", type = str, help = "Path to output directory.")
    #xparser.add_argument("config_yml", nargs = 1, metavar = "config_yml", type = str, help = "Path to configuration dataset YML file.")
    #parser.add_argument("ref_yml", nargs = 1, metavar = "ref_yml", type = str, help = "Path to reference dataset YML file.")
    #parser.add_argument("sim_yml", nargs = 1, metavar = "sim_yml", type = str, help = "Path to simulation dataset YML file.")
    parser.add_argument("-c", nargs = 1, metavar = "--config", type = str, help = "Optional path to YML file to override default variable configurations.")

    args = parser.parse_args()
    var_configs = args.c
    print(args)
    from pathlib import Path
    plot_loc = Path(args.output_dir[0])
    if not plot_loc.is_dir():
        print(f"\tINFO: Directory not found, making new plot save location")
        plot_loc.mkdir(parents=True)
    
    if args.c is None:
        var_configs = PATH_VARIABLE_DEFAULTS
    else:
        var_configs = args.c[0]

    from pathlib import Path
    
    #save_loc.mkdir(parents=True, exist_ok=True)
    #from io import get_input_data
    from file_io import get_input_data
    #from cvdp.io import get_input_data
    print("PARENT_DIR",PARENT_DIR,"\n")




    """
    #def mean_seasonal_calc(ds_name, dataset, var_name, config_dict):
    save_loc = config_dict[ds_name]["save_loc"]
    Path(save_loc).mkdir(parents=True, exist_ok=True)
    syr = config_dict[ds_name]["syr"]
    eyr = config_dict[ds_name]["eyr"]

    ts_filename = f'{ds_name}.cvdp_data.{var_name}.climo.ts.{syr}-{eyr}.nc'
    ts_fno = save_loc / Path(ts_filename)

    data_dict = {}
    calc_all_mean = False
    #if ts_fno.is_file():
    if "members" in config_dict[ds_name]:
        print("AtmOcnMean.py:  members are in this case:",ds_name)

        if ts_fno.is_file():
            seas_ts = xr.open_dataset(ts_fno)
        members = config_dict[ds_name]["members"]
        for member in members:
            ts_mem_filename = f'{ds_name}.cvdp_data.{var_name}{member}climo.ts.{syr}-{eyr}.nc'
            ts_mem_fno = save_loc / Path(ts_mem_filename)

            ts_mem_mean_filename = f'{ds_name}.cvdp_data.{var_name}{member}climo.ts.mean.{syr}-{eyr}.nc'
            ts_mem_mean_fno = save_loc / Path(ts_mem_mean_filename)

            if ts_mem_fno.is_file() and ts_mem_mean_fno.is_file():
                print(f"\tFound pre-existing climatology files for {ds_name} {var_name} {member}, loading from disk...\n")
                seas_mem_ts = xr.open_dataset(ts_mem_fno)
                data_dict[f"seas_ts{member[:-1]}"] = seas_mem_ts

                seas_mem_mean_ts = xr.open_dataset(ts_mem_mean_fno)
                data_dict[f"seas_ts{member[:-1]}_mean"] = seas_mem_mean_ts
                calc_all_mean = False

        # Average all members if applicable    
        if calc_all_mean:
            #seas_ts = xr.open_dataset(ts_fno)
            seas_ts = seas_ts.mean(dim="member", keep_attrs=True)
            seas_ts.attrs["members"] = members
            seas_ts.to_netcdf(ts_fno)
            print(f"\tSUCCESS: Climatological seasonal mean over members saved to file: {ts_fno}\n")
        #else:
        #    seas_ts = xr.open_dataset(ts_fno)
    else:
        print("AtmOcnMean.py:  members are NOT in this case:",ds_name)
        if ts_fno.is_file():
            print(f"\tFound pre-existing climatology files for {ds_name} {var_name}, loading from disk...\n")
            seas_ts = xr.open_dataset(ts_fno)

    """






    if "c" not in args:
        # These are dictionaries of datasets
        ref_datasets, sim_datasets, config_dict = get_input_data(f"{PARENT_DIR}/example_config.yaml")
    else:
        ref_datasets, sim_datasets, config_dict = get_input_data(f"{args.c[0]}")
    print(list(ref_datasets.keys()))
    ref_names = list(ref_datasets.keys())
    sim_names = list(sim_datasets.keys())
    print("Reference Names:",ref_names)
    print("Simulation Names:",sim_names,"\n")

    vns = ["psl"]
    for vn in vns:
        kwargs = {}
        #print(f"\nProcessing variable: {vn}\n")
        for ref_name in ref_names:
            print(f"Trying reference {ref_name} for climatologies")
            if ref_name not in kwargs:
                kwargs[ref_name] = {}
            #kwargs[ref_name]["run_type"] = "reference"
            kwargs[f"{ref_name}_run_type"] = "reference"
            ref_var = ref_datasets[ref_name][vn]
            data_dict = mean_seasonal_calc(ref_name, ref_var,
                                               vn, config_dict)
            kwargs[f"{ref_name}_season_trnd_avgs"] = ref_var
            ref_seas_ts = data_dict["seas_ts"]
            kwargs[ref_name] = ref_seas_ts
            if "members" in ref_seas_ts.attrs:
                members = ref_seas_ts.attrs["members"]
                kwargs[ref_name]["members"] = members
                for member in members:
                    kwargs[f"{ref_name}{member[:-1]}"] = data_dict[f"seas_ts{member[:-1]}"]
        for sim_name in sim_names:
            print(f"Trying simulation {sim_name} for climatologies")
            if sim_name not in kwargs:
                kwargs[sim_name] = {}
            kwargs[f"{sim_name}_run_type"] = "simulation"
            sim_var = sim_datasets[sim_name][vn]
            data_dict = mean_seasonal_calc(sim_name, sim_var,
                                               vn, config_dict)
            kwargs[f"{sim_name}_season_trnd_avgs"] = sim_var
            sim_seas_ts = data_dict["seas_ts"]
            kwargs[sim_name] = sim_seas_ts

            if "members" in sim_seas_ts.attrs:
                members_sub = config_dict[sim_name]["members"]
                members = sim_seas_ts.attrs["members"]
                huh = [mem for mem in members if mem in members_sub]
                kwargs[f"{sim_name}_members"] = huh
                for member in huh:
                    try:
                        kwargs[f"{sim_name}{member[:-1]}"] = data_dict[f"seas_ts{member[:-1]}"]
                        kwargs[f"{sim_name}{member[:-1]}_trnds"] = sim_var.sel(member=member)
                    except KeyError as e:
                        print(f"seas_ts{member[:-1]}")

        kwargs["ref_seas_ts"] = ref_seas_ts
        kwargs["sim_seas_ts"] = sim_seas_ts
        #kwargs["ref_season_trnd_avgs"] = ref_season_trnd_avgs
        #kwargs["sim_season_trnd_avgs"] = sim_season_trnd_avgs
        kwargs["vn"] = vn
        kwargs["sim_names"] = sim_names
        kwargs["ref_names"] = ref_names
        graphics(plot_loc, **kwargs)

if __name__ == '__main__':
    main()