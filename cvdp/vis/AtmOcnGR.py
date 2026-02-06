#!/usr/bin/env python3
"""
AtmOcnGR.py

CVDP functions for plotting climatological diagnostics:
means, standard deviations, trends, timeseries, metrics tables, etc.

License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import xskillscore as xs

from vis.global_plots import (
    global_ensemble_plot,
    global_indmem_latlon_plot,
    global_indmemdiff_latlon_plot,
)
from vis.polar_plots import (
    polar_ensemble_plot,
    polar_indmem_latlon_plot,
    polar_indmemdiff_latlon_plot,
)
from vis.timeseries_plot import timeseries_plot

import cvdp_utils.avg_functions as af
import cvdp_utils.utils as helper_utils
import cvdp_utils.analysis as an

SEASON_LIST = ["DJF","SON"]
SEASON_LIST = ["DJF"]
VAR_SEASONS = {
    "psl": {"global": SEASON_LIST + ["NDJFM"], "polar": SEASON_LIST, "timeseries": SEASON_LIST},
    "ts": SEASON_LIST,
    "trefht": SEASON_LIST,
    "prect": SEASON_LIST,
}

EOF_VARS = ["NAM", "SAM", "PSA1", "PSA2"]
NH_VARS = ["NAM"]
SH_VARS = ["SAM", "PSA1", "PSA2"]

ANLYS_TYPES = ["spatialmean", "trends", "spatialstddev"]
#ANLYS_TYPES = ["spatialstddev"]
#ANLYS_TYPES = ["spatialmean", "trends"]
#ANLYS_TYPES = ["trends"]
#ANLYS_TYPES = ["spatialmean"]

MAP_TYPES = ["global", "polar", "timeseries"]
MAP_TYPES = ["polar", "timeseries"]
MAP_TYPES = ["polar"]
MAP_TYPES = ["timeseries"]
MAP_TYPES = ["global"]

PLOT_TYPES = ["summary", "indmem", "indmemdiff"]
#PLOT_TYPES = ["summary"]
#PLOT_TYPES = ["summary","indmem"]


def get_plot_title(var, plot_type, ptype, season):
    if ptype == "trends" and var in ["NPI"] + EOF_VARS:
        ptype = "Pattern"
    base = f"{var} {ptype.capitalize()} ({season})"
    titles = {
        "summary": f"Ensemble Summary: {base}",
        "indmem": f"{base}\n",
        "indmemdiff": f"{base} Differences\n",
    }
    return titles.get(plot_type, "Unknown Title")


def get_plot_name(vn, var, ptype, season, plot_type, map_type):
    season_lower = season.lower()
    suffix = f"{ptype}_{season_lower}"

    if ptype == "trends" and var in ["NPI"] + EOF_VARS:
        suffix = "pattern_" + season_lower
        vn = var.lower()
        if map_type == "timeseries":
            suffix = f"{map_type}_{season_lower}"

    return f"{vn}_{suffix}.{plot_type}.png"


def compute_diff(sim, ref):
    interp = an.interp_diff(sim, ref)
    return sim - (interp if interp is not None else ref)


def compute_npi(sim_ts, ref_ts):
    def _standardize(arr):
        return (arr - arr.mean("time")) / arr.std("time")

    def _npi(arr):
        sliced = arr.sel(lat=slice(30, 65), lon=slice(160, 220))
        weighted = sliced.weighted(np.cos(np.radians(sliced.lat))).mean(["lat", "lon"])
        return _standardize(weighted)

    sim_npi = xs.linslope(_npi(sim_ts), sim_ts, dim="time")
    ref_npi = xs.linslope(_npi(ref_ts), ref_ts, dim="time")
    return sim_npi, ref_npi, compute_diff(sim_npi, ref_npi)


def compute_eof(var, sim_anom, ref_anom, season):
    index_map = {"NAM": 0, "SAM": 0, "PSA1": 1, "PSA2": 2}
    num = index_map[var]
    bounds = {"n": 90, "s": 20} if var in NH_VARS else {"n": -20, "s": -90}

    sim_attrs = sim_anom.attrs.copy()
    ref_attrs = ref_anom.attrs.copy() 

    def _eof(arr, invert=False):
        eofs, pcs, slp = an.get_eof(arr, season, bounds, neof=3)
        pcs_std = (pcs.sel(pc=num) - pcs.sel(pc=num).mean("time")) / pcs.sel(pc=num).std("time")
        return xs.linslope(-pcs_std if invert else pcs_std, slp, dim="time"), pcs_std

    sim_pattern, sim_pc = _eof(sim_anom, var == "SAM")
    ref_pattern, ref_pc = _eof(ref_anom, var == "PSA2")
    sim_pattern.attrs = sim_attrs
    ref_pattern.attrs = ref_attrs

    diff_pattern = compute_diff(sim_pattern, ref_pattern)
    diff_pattern.attrs['units'] = sim_pattern.units
    return sim_pattern, ref_pattern, diff_pattern, sim_pc, ref_pc


def plot_dispatch(plot_type, ptype, map_type, vn, var, sims, refs, diffs, vtres, title, sims_ens=None, refs_ens=None, pcs=None):
    if map_type == "timeseries" and pcs:
        print(timeseries_plot(var, pcs[0], pcs[1]))
        return timeseries_plot(var, pcs[0], pcs[1])

    if plot_type == "summary":
        if map_type == "global":
            return global_ensemble_plot([sims_ens, refs_ens], diffs, vn, ptype, vtres, title)
        if map_type == "polar":
            return polar_ensemble_plot([sims_ens, refs_ens], diffs, vn, var, ptype, vtres, title)

    elif plot_type == "indmem":
        if map_type == "global":
            #print("asdfsadfsdfsdfsdf",len(sims),len(refs),"\n")
            return global_indmem_latlon_plot(vn, [sims, refs], vtres, title, ptype)
        if map_type == "polar":
            return polar_indmem_latlon_plot(vn, var, [sims, refs], vtres, title, ptype)

    elif plot_type == "indmemdiff":
        #runs = f"{sims.run} - {refs.run}"
        runs = []
        for sim in sims:
            for ref in refs:
                runs.append(f"{sim.run} - {ref.run}")
        if map_type == "global":
            return global_indmemdiff_latlon_plot(vn, runs, diffs, ptype, vtres, title)
        if map_type == "polar":
            return polar_indmemdiff_latlon_plot(vn, var, runs, diffs, ptype, vtres, title)

    return None


def graphics(plot_loc, **kwargs):
    print("\n\n\n\n\n\nkwargs", kwargs.keys()),"\n\n\n\n"
    res = helper_utils.get_variable_defaults()
    vn = kwargs["vn"]
    sim_names = kwargs["sim_names"]
    ref_names = kwargs["ref_names"]
    dont_continue = False
    for ptype in ANLYS_TYPES:
        print(f"*** Analysis Type: {ptype}")
        for map_type in MAP_TYPES:
            print(f"  *** Map Type: {map_type}")
            seasons = VAR_SEASONS[vn][map_type] if isinstance(VAR_SEASONS[vn], dict) else VAR_SEASONS[vn]
            for season in seasons:
                for plot_type in PLOT_TYPES:
                    print(f"    *** Plot Type: {plot_type}")
                    print("\t ", vn, ptype, map_type, plot_type, season)
                    key = f"{vn}_{ptype}_{season.lower()}"

                    figs = []
                    names = []

                    # NPI case
                    if ptype == "trends" and vn == "psl" and map_type == "global" and season == "NDJFM":
                        var = "NPI"
                        vres = res[var][ptype]
                        sim_npi = kwargs["sim_seas_ts"][key]
                        ref_npi = kwargs["ref_seas_ts"][key]
                        sim_attrs = sim_npi.attrs.copy()
                        ref_attrs = ref_npi.attrs.copy()
                        
                        sim_npi, ref_npi, diff_npi = compute_npi(sim_npi,ref_npi)
                        
                        sim_npi.attrs = sim_attrs
                        ref_npi.attrs = ref_attrs
                        diff_npi.attrs["units"] = sim_attrs["units"]
                        
                        title = get_plot_title(var, plot_type, ptype, season)
                        name = get_plot_name(vn, var, ptype, season, plot_type, map_type)
                        fig = plot_dispatch(plot_type, ptype, map_type, vn, var, sim_npi, ref_npi, diff_npi, vres, title)
                        if fig:
                            figs.append((fig, name))
                    #print("\t  Season: ", season)
                    """
                    # EOF case
                    elif ptype == "trends" and vn == "psl" and map_type in ["polar", "timeseries"]:
                        for var in EOF_VARS:
                            vres = res[var][ptype]
                            sim_data = kwargs["sim_season_trnd_avgs"]
                            #print("DVDVDVDFVD",sim_data,"\n====================\n")
                            ref_data = kwargs["ref_season_trnd_avgs"]
                            #print("seas_ts.psl_spatialmean_djf.attrs",sim_data.attrs,"\n    mpl, p,l, [l, [;., [;.] ]. ]['. ]'/ ]'\n")
                            if "member" in sim_data.coords:
                                attrs = sim_data.attrs  # save before doing groupby/mean
                                members = sim_data.member
                                #print("members",members)
                                sim_data = sim_data.mean(dim="member", keep_attrs=True)
                                sim_data.attrs = attrs
                                sim_data.attrs["members"] = members.values
                            if "member" in ref_data.coords:
                                attrs = ref_data.attrs  # save before doing groupby/mean
                                members = ref_data.member
                                #print("members",members)
                                ref_data = ref_data.mean(dim="member", keep_attrs=True)
                                ref_data.attrs = attrs
                                ref_data.attrs["members"] = members.values
                            sim, ref, diff, sim_pc, ref_pc = compute_eof(
                                var,
                                sim_data, #kwargs["sim_season_trnd_avgs"],
                                ref_data, #kwargs["ref_season_trnd_avgs"],
                                season,
                            )
                            title = get_plot_title(var, plot_type, ptype, season)
                            name = get_plot_name(vn, var, ptype, season, plot_type, map_type)
                            fig = plot_dispatch(plot_type, ptype, map_type, vn, var, sim, ref, diff, vres, title, pcs=(sim_pc, ref_pc))
                            if fig:
                                figs.append((fig, name))
                    """
                    # Standard seasonal diagnostics
                    elif season != "NDJFM":
                        if map_type == "polar":
                            print("Skipping polar plot for non EOF vars")
                            dont_continue = True
                            continue
                        vres = res[vn][ptype]
                        #sim_attrs = sim_data.attrs.copy()
                        #ref_attrs = ref_data.attrs.copy()

                        """
                        #if dont_continue:
                        if ptype == "spatialstddev":
                            sim = sim_data
                            ref = ref_data
                        elif ptype == "trends":
                            #print("sim_data before linregersss",sim_data,"\n====================\n")
                            sim = af.lin_regress(sim_data)[0]
                            #print("sim",sim,"\n====================\n")
                            ref = af.lin_regress(ref_data)[0]
                        else:
                        """
                        if 1==1:
                            sims = []
                            sims_ens = []
                            for sim_name in sim_names:
                                print("\t     Processessing simulation run: ",sim_name)
                                sim_dataset = kwargs[f"{sim_name}"]
                                sim_data = sim_dataset[key]
                                sim_attrs = sim_data.attrs.copy()
                                #print("sim_dataset",sim_dataset.attrs)
                                if "members" in sim_dataset.attrs:
                                    members = sim_dataset.attrs["members"]
                                    for member in members:
                                        print("\t        Processessing simulation member: ",member)
                                        sim_dataset_m = kwargs[f"{sim_name}{member[:-1]}"]
                                        sim_data = sim_dataset_m[key]
                                        if "time" in sim_data.dims:
                                            sim = sim_data.mean("time")
                                        else:
                                            sim = sim_data
                                        #sim = sim_data.mean("time")
                                        sim.attrs = sim_dataset.attrs
                                        sims.append(sim)
                                    if "time" in sim_dataset[key].dims:
                                        sim_ug = sim_dataset[key].mean("time")
                                    else:
                                        sim_ug = sim_dataset[key]
                                    sim_ug.attrs = sim_attrs
                                    sim_ug.attrs["members"] = members
                                    #print("sim_ug.attrs",sim_ug.attrs,"\n-------------------------n")
                                    #sim_ug.attrs {'units': 'hPa', 'long_name': 'psl spatialmean DJF', 'run': 'CESM2-LE', 'yrs': [np.int64(1979), np.int64(2013)],
                                    # 'members': array(['.001.', '.002.', '.003.'], dtype='<U5')}
                                    sims_ens.append(sim_ug)
                                else:
                                    print("sim_data",sim_data.dims)
                                    if "time" in sim_data.dims:
                                        sim = sim_data.mean("time")
                                    else:
                                        sim = sim_data
                                    sim.attrs = sim_dataset.attrs
                                    sims_ens.append(sim)
                                    sims.append(sim)


                            refs = []
                            refs_ens = []
                            for ref_name in ref_names:
                                print("\t     Processessing reference run: ",ref_name)
                                ref_dataset = kwargs[f"{ref_name}"]
                                ref_data = ref_dataset[key]
                                ref_attrs = ref_data.attrs.copy()
                                if "members" in ref_attrs:
                                    members = ref_attrs["members"]
                                    for member in members:
                                        print("\t        Processessing reference member: ",member)
                                        ref_dataset = kwargs[f"{ref_name}{member[:-1]}"]
                                        ref_data = ref_dataset[key]
                                        if "time" in ref_data.dims:
                                            ref = ref_data.mean("time")
                                        else:
                                            ref = ref_data
                                        ref.attrs = ref_attrs
                                        refs.append(ref)
                                    refs_ens.append(ref_data.mean("time"))
                                else:
                                    #ref = ref_data.mean("time")
                                    print("ref_data",ref_data.dims)
                                    if "time" in ref_data.dims:
                                        ref = ref_data.mean("time")
                                    else:
                                        print("No time dimensions?")
                                        ref = ref_data
                                    ref.attrs = ref_attrs
                                    refs_ens.append(ref)
                                    refs.append(ref)
                        #sim.attrs = sim_attrs
                        #ref.attrs = ref_attrs
                        #print("sim.attrs", sim.attrs)
                        #print("ref.attrs", ref.attrs)
                        #if not dont_continue:
                        diffs = []
                        for simel in sims:
                            for refel in refs: 
                                diff = compute_diff(simel, refel)
                                diff.attrs["units"] = sim_attrs.get("units")
                                diffs.append(diff)
                        title = get_plot_title(vn.upper(), plot_type, ptype, season)
                        name = get_plot_name(vn, vn, ptype, season, plot_type, map_type)

                        fig = plot_dispatch(plot_type, ptype, map_type, vn, vn, sims, refs, diffs, vres, title, sims_ens=sims_ens, refs_ens=refs_ens, pcs=None)
                        names.append(name)
                        if fig:
                            figs.append((fig, name))

                    print("FIGS",figs)
                    print(names)
                    # Save figures
                    for fig, name in figs:
                        print("NAME",name)
                        fig.savefig(plot_loc / name, bbox_inches="tight", dpi=150)
                        plt.close(fig)
            print(f"  Map Type End ***")
        print(f"Analysis Type End ***\n\n")