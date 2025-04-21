import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import plotting.plot_dicts as plot_dicts
import plotting.ncl_colormaps as nclcmaps
import plotting.ncl_masks as ncl_masks
from plotting.utils import clean_data


def wrap_and_clean_data(arr, vn, ptype, diff=False):
    lon_idx = arr.dims.index("lon")
    wrap_data, wrap_lon = add_cyclic_point(arr.values, coord=arr.lon, axis=lon_idx)
    wrap_data = clean_data(vn, wrap_data, ptype, diff=diff)
    return wrap_data, wrap_lon, arr.lat


def apply_land_mask_if_ts(ax, vn, cmap, levels):
    if vn != "ts":
        return None

    land_mask = ncl_masks.LSMASK.where(ncl_masks.LSMASK == 1)
    lon_idx = land_mask.dims.index("lon")
    wrap_data_land, wrap_lon_land = add_cyclic_point(land_mask.values, coord=land_mask.lon, axis=lon_idx)
    norm = mpl.colors.BoundaryNorm(levels, cmap.N)
    ax.contourf(wrap_lon_land, land_mask.lat, wrap_data_land, colors="w", zorder=300, transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAKES.with_scale("110m"), edgecolor="#b5b5b5", facecolor="none", zorder=300)
    return norm


def create_colorbar(fig, ax, img, levels, cbarticks, unit, long_label=False):
    axins = inset_axes(ax, width="85%" if not long_label else "100%", height="5%", loc='lower center', borderpad=-5)
    cb = fig.colorbar(img, orientation='horizontal', cax=axins, ticks=levels, extend='both')
    tick_labels = [str(int(t)) if t in cbarticks else '' for t in levels]
    cb.set_ticklabels(tick_labels)
    cb.ax.set_xlabel(unit, fontsize=18)
    cb.ax.tick_params(labelsize=16, size=0)
    cb.outline.set_visible(False)


def plot_panel(ax, arr, vn, ptype, levels, cmap, cbarticks, unit, title_text, show_colorbar=False, fig=None, long_label=False, diff=False):
    wrap_data, wrap_lon, lat = wrap_and_clean_data(arr, vn, ptype, diff=diff)
    norm = apply_land_mask_if_ts(ax, vn, cmap, levels)

    img = ax.contourf(wrap_lon, lat, wrap_data, cmap=cmap, levels=levels,
                      transform=ccrs.PlateCarree(), norm=norm)
    
    ax.set_title(title_text, fontsize=18, color="#0c80ab")
    ax.coastlines("50m", color="#b5b5b5")

    if show_colorbar and fig:
        create_colorbar(fig, ax, img, levels, cbarticks, unit, long_label=long_label)

    return img



def global_enesmble_plot(arrs, arr_diff, vn, season, ptype, plot_dict, title, debug=False):
    fig, axs = plt.subplots(1, 4, figsize=(25, 15),
                            subplot_kw={"projection": ccrs.Robinson(central_longitude=210)})

    for r, ax in enumerate(axs):
        if r == 2:
            arr = arr_diff.sel(season=season)
            label = "Difference"
            cmap = plot_dicts.diff[vn][ptype]["cmap"]
            levels = plot_dicts.diff[vn][ptype]["levels"]
            cbarticks = plot_dicts.diff[vn][ptype]["cbarticks"]
            unit = plot_dicts.units[vn]
            diff_flag = True
        elif r == 3:
            arr = arrs[0] * 0  # dummy array for rank panel
            label = "Rank"
            cmap = "Greys"
            levels = [-1, 1]
            cbarticks = [-1, 1]
            unit = ""
            diff_flag = False
        else:
            arr = arrs[r].sel(season=season)
            label = getattr(arr, "run_name", f"Case {r+1}")
            cmap = plot_dicts.cmap[vn][ptype]
            levels = plot_dicts.levels[vn][ptype]
            cbarticks = plot_dicts.cbarticks[vn][ptype]
            unit = plot_dicts.units[vn]
            diff_flag = False

        plot_panel(ax, arr, vn, ptype, levels, cmap, cbarticks, unit, label,
                   show_colorbar=True, fig=fig, diff=diff_flag)

    fig.text(0.92, 0.61, "$\\copyright$ CVDP-LE", fontsize=10,
             color='#b5b5b5', weight='bold', alpha=0.75, ha='right', va='top')
    plt.suptitle(title, fontsize=24, y=0.63)
    plt.subplots_adjust(wspace=0.1)

    return fig


def global_sim_ref_plot(sim, ref, vn, season, ptype, plot_dict, title, debug=False):
    fig, axs = plt.subplots(2, 1, figsize=(12, 15),
                            subplot_kw={"projection": ccrs.Robinson(central_longitude=210)})

    for i, (arr, label) in enumerate(zip([sim, ref], ["Simulation", "Reference"])):
        arr_season = arr.sel(season=season)
        cmap = plot_dicts.cmap[vn][ptype]
        levels = plot_dicts.levels[vn][ptype]
        cbarticks = plot_dicts.cbarticks[vn][ptype]
        unit = plot_dicts.units[vn]

        plot_panel(axs[i], arr_season, vn, ptype, levels, cmap, cbarticks, unit, label,
                   show_colorbar=True, fig=fig)

    fig.text(0.98, 0.05, "$\\copyright$ CVDP-LE", fontsize=10,
             color='#b5b5b5', weight='bold', alpha=0.75, ha='right', va='bottom')
    plt.suptitle(title, fontsize=24, y=0.95)

    return fig


def global_diff_plot(diff_arr, vn, season, ptype, plot_dict, title, debug=False):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6),
                           subplot_kw={"projection": ccrs.Robinson(central_longitude=210)})

    arr = diff_arr.sel(season=season)
    cmap = plot_dicts.diff[vn][ptype]["cmap"]
    levels = plot_dicts.diff[vn][ptype]["levels"]
    cbarticks = plot_dicts.diff[vn][ptype]["cbarticks"]
    unit = plot_dicts.units[vn]

    plot_panel(ax, arr, vn, ptype, levels, cmap, cbarticks, unit, "Difference",
               show_colorbar=True, fig=fig, diff=True)

    fig.text(0.98, 0.05, "$\\copyright$ CVDP-LE", fontsize=10,
             color='#b5b5b5', weight='bold', alpha=0.75, ha='right', va='bottom')
    plt.suptitle(title, fontsize=24, y=0.90)

    return fig
