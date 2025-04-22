#!/usr/bin/env python3
"""
seasonal_plots.py

Creates plots for seasonal climatology metrics.
License: MIT
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cartopy.util import add_cyclic_point

from vis import *
from visualization.vis_utils import *
import old_utils.avg_functions as af

# Load masks once
lsmask, ncl_masks = af.land_mask()


def get_plot_config(r, arrs, arr_diff, season, plot_dict, vn):
    """Return data, title, levels, colormap, and other config for each subplot."""
    unit = arrs[0].units
    if r == 3:
        arr = af.zeros_array(arrs[-1].shape[0], arrs[-1].shape[1])
        title = "Rank of Observations within Ensemble"
        cmap = bg_cmap
        levels = [-5, 0, 5, 10, 20, 80, 90, 95, 100, 105]
        norm = PiecewiseNorm([0, 5, 10, 20, 80, 90, 95, 100])
        unit = "%"
        yrs_text = ''
    elif r == 2:
        arr = arr_diff.sel(season=season)
        title = f"{arrs[0].run_name} - {arrs[1].run_name}"
        diff_range = plot_dict.get("diff_range")
        levels = np.arange(*diff_range) if diff_range else np.linspace(arr.min().item(), arr.max().item(), 20)
        ticks = np.arange(*plot_dict.get("diff_ticks_range", (levels[0], levels[-1], (levels[-1] - levels[0]) / 10)))
        cbarticks = plot_dict.get("diff_cbarticks", ticks)
        cmap = plot_dict.get("diff_cmap", plot_dict["cmap"])
        cmap = cmap if cmap in plt.colormaps() else get_NCL_colormap(cmap)
        norm = None
        yrs_text = ''
    else:
        arr = arrs[r].sel(season=season)
        title = arr.run_name
        levels = np.linspace(*plot_dict["contour_levels_linspace"])
        ticks = np.arange(*plot_dict["ticks_range"])
        cbarticks = plot_dict.get("cbarticks", ticks)
        cmap = plot_dict["cmap"]
        cmap = cmap if cmap in plt.colormaps() else get_NCL_colormap(cmap)
        norm = mpl.colors.BoundaryNorm(ticks, amwg_cmap.N) if vn == "ts" else None
        yrs_text = f'{arr.yrs[0]}-{arr.yrs[1]}'

    return arr, title, cmap, levels, ticks, cbarticks, norm, unit, yrs_text


def plot_field(ax, arr, levels, cmap, norm, vn, ptype):
    """Plot the data on a given Axes with land masking and coastlines."""
    lat = arr.lat
    lon_idx = arr.dims.index('lon')
    wrap_data, wrap_lon = add_cyclic_point(arr.values, coord=arr.lon, axis=lon_idx)
    wrap_data = clean_data(vn, wrap_data, ptype, diff=False)

    img = ax.contourf(wrap_lon, lat, wrap_data, levels=levels, cmap=cmap,
                      norm=norm, transform=ccrs.PlateCarree())

    # Optional land mask overlay
    if vn == "ts":
        landsies = ncl_masks.LSMASK.where(ncl_masks.LSMASK == 1)
        wrap_data_land, wrap_lon_land = add_cyclic_point(
            landsies.values,
            coord=landsies.lon,
            axis=landsies.dims.index('lon')
        )
        ax.contourf(wrap_lon_land, landsies.lat, wrap_data_land,
                    colors="w", transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAKES.with_scale('110m'),
                       edgecolor="#b5b5b5", facecolor="none", zorder=300)

    ax.coastlines(color="#b5b5b5")
    return img


def add_colorbar(fig, ax, img, unit, ticks, cbarticks):
    """Add a custom horizontal colorbar below the plot."""
    axins = inset_axes(ax, width="85%", height="8%", loc='lower center', borderpad=-3)
    cb = fig.colorbar(img, orientation='horizontal', cax=axins, ticks=ticks, extend='both')
    tick_labels = [str(int(loc)) if loc in cbarticks else '' for loc in ticks]
    cb.set_ticklabels(tick_labels)
    cb.ax.set_xlabel(unit, fontsize=18)
    cb.ax.tick_params(labelsize=12, size=0)
    cb.outline.set_visible(False)


def global_ensemble_plot(arrs, arr_diff, vn, season, ptype, plot_dict, title, debug=False):
    """
    Create a 4-panel ensemble plot.
    """
    nrows, ncols = 1, 4
    fig_width, fig_height = 15 + 2.5 * ncols, 15
    proj = ccrs.Robinson(central_longitude=210)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height),
                            facecolor='w', edgecolor='k', sharex=True, sharey=True,
                            subplot_kw={"projection": proj})
    
    for r in range(ncols):
        arr, run_title, cmap, levels, ticks, cbarticks, norm, unit, yrs_text = get_plot_config(
            r, arrs, arr_diff, season, plot_dict, vn
        )

        if debug:
            print(f"[DEBUG] Panel {r} — Title: {run_title}, Years: {yrs_text}, Unit: {unit}")

        img = plot_field(axs[r], arr, levels, cmap, norm, vn, ptype)
        #axs[r].set_title(run_title, fontsize=18, color='#0c80ab' if r == 0 else None)
        if r == 0:
            axs[r].set_title(run_title, fontsize=18, color='#0c80ab')
        else:
            axs[r].set_title(run_title, fontsize=18)
        axs[r].text(-0.065, 0.98, yrs_text, transform=axs[r].transAxes, fontsize=11, va='top')

        add_colorbar(fig, axs[r], img, unit, ticks, cbarticks)

    # Add extra text and layout formatting
    axs[0].text(.875, 0.98, 'r=0.28', transform=axs[0].transAxes, fontsize=11, verticalalignment='top')
    axs[-1].text(.875, 0.99, "--%", transform=axs[-1].transAxes, fontsize=12, verticalalignment='top')
    fig.text(0.92, 0.61, "$\\copyright$ CVDP-LE",
             fontsize=10, color='#b5b5b5', weight='bold', alpha=0.75,
             ha='right', va='top')

    plt.suptitle(title, fontsize=24, y=0.63)
    plt.subplots_adjust(wspace=0.1)

    return fig
