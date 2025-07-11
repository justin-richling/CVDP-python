#!/usr/bin/env python3
"""
polar_plots_refactored.py

A tidier, more modular re‑write of the original *polar_plots.py*.
The three user‑facing functions are kept (with the **same** call signatures)
so existing code does not break, but the heavy lifting is delegated
into small, reusable helpers so maintenance is easier and you can plug
additional plots in with minimal copy‑paste.

© 2025 — MIT licence (same as original)
"""
from __future__ import annotations

from typing import Iterable, Tuple, Sequence
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.path as mpath
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from vis import *  # noqa: F401, pylint: disable=wildcard-import
from visualization.vis_utils import *  # noqa: F401, pylint: disable=wildcard-import
import old_utils.avg_functions as af

lsmask, ncl_masks = af.land_mask()

# ============================================================================
# ------------------------------ Helper API ----------------------------------
# ============================================================================

def _get_projection(var: str) -> Tuple[ccrs.CRS, Sequence[float], int]:
    """Return the polar projection, map extent and y‑offset for lon labels."""
    nh_vars = {"NAM", "PNO", "PNA"}
    sh_vars = {"SAM", "PSA1", "PSA2"}

    if var in nh_vars:
        return (ccrs.NorthPolarStereo(central_longitude=0), [-180, 180, 20, 90], 16)
    if var in sh_vars:
        return (ccrs.SouthPolarStereo(central_longitude=0), [-180, 180, -20, -90], -16)

    raise ValueError(f"Unknown polar variable: {var}")


def _lon_labels(ax, space: int, fontsize: int = 14) -> None:
    """Draw longitude labels around a polar plot."""
    lon_ticks = np.arange(0, 360, 30)
    lon_labels = [
        "0", "30E", "60E", "90E", "120E", "150E",
        "180", "150W", "120W", "90W", "60W", "30W",
    ]
    for lon, label in zip(lon_ticks, lon_labels):
        x, y = ccrs.PlateCarree().transform_point(lon, space, ccrs.PlateCarree())
        ax.text(x, y, label, transform=ccrs.PlateCarree(), fontsize=fontsize,
                ha="center", va="center")


def _polar_axis(ax: plt.Axes, extent: Sequence[float], space: int) -> None:
    """Set common formatting for a polar axis."""
    # Circular boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    circle = mpath.Path(np.column_stack([np.sin(theta), np.cos(theta)]) * 0.5 + 0.5)

    ax.set_extent(extent, ccrs.PlateCarree())
    ax.set_boundary(circle, transform=ax.transAxes)
    _lon_labels(ax, space)

    # Cosmetic details
    plt.setp(ax.spines.values(), lw=0.5, color="grey", alpha=0.7)
    ax.coastlines("50m", color="#b5b5b5", alpha=0.5)


def _levels_ticks(plot_info: dict, *, diff: bool, arr: "xr.DataArray | None" = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Figure out contour *levels*, plotted *ticks*, and *cbarticks*.

    When nothing explicit is requested in *plot_info*, the data are inspected.
    """
    # --- 1. Choose the set of levels -------------------------------------------------
    levels: np.ndarray | None = None

    if not diff:  # regular plot
        if "contour_levels_linspace" in plot_info:
            levels = np.linspace(*plot_info["contour_levels_linspace"])
        elif "contour_levels_range" in plot_info:
            levels = np.arange(*plot_info["contour_levels_range"])
        elif "contour_levels_list" in plot_info:
            levels = np.asarray(plot_info["contour_levels_list"], dtype=float)
    else:  # difference plot
        if "diff_range_list" in plot_info:
            levels = np.asarray(plot_info["diff_range_list"], dtype=float)

    if (levels is None or not isinstance(levels, np.ndarray)) and arr is not None:
        # fall back to data min/max
        vmin, vmax = float(arr.min()), float(arr.max())
        levels = np.linspace(vmin, vmax, 20)

    # --- 2. Ticks and cbarticks ------------------------------------------------------
    tick_key = "diff_ticks_range" if diff else "ticks_range"
    ticks = np.arange(*plot_info[tick_key]) if tick_key in plot_info else levels

    cbar_key = "diff_cbarticks_range" if diff else "cbarticks"
    cbarticks = np.asarray(plot_info.get(cbar_key, ticks))

    return np.asarray(levels), np.asarray(ticks), np.asarray(cbarticks)


def _colormap(name: str) -> mpl.colors.Colormap:
    """Return a Matplotlib (or NCL) colormap by *name*, silently falling back."""
    if name in plt.colormaps():
        return plt.get_cmap(name)
    # build from NCL helper if necessary (supplied by vis_utils)
    return get_NCL_colormap(name, extend="None")


def _add_colorbar(fig: plt.Figure, img: mpl.contour.QuadContourSet, *, unit: str,
                  levels: np.ndarray, ticks: np.ndarray, cbarticks: np.ndarray,
                  ax_anchor: plt.Axes | Sequence[plt.Axes], fontsize: int = 16) -> None:
    """Create a horizontal inset colourbar underneath *ax_anchor*."""
    # Use the last axis in a sequence to place the bar underneath stacked plots
    anchor = ax_anchor[-1] if isinstance(ax_anchor, Sequence) else ax_anchor
    cax = inset_axes(anchor, width="120%", height="5%", loc="lower center", borderpad=-5)

    cb = fig.colorbar(img, orientation="horizontal", cax=cax, ticks=ticks, extend="both")

    # Tick labels only for cbarticks
    label_map = {val: (f"{int(val):d}" if val.is_integer() else f"{val}") for val in cbarticks}
    tick_labels = [label_map.get(t, "") for t in ticks]
    cb.set_ticklabels(tick_labels)

    # Style
    cb.ax.set_xlabel(unit, fontsize=fontsize + 2)
    cb.ax.tick_params(labelsize=fontsize, size=0)
    cb.outline.set_visible(False)

# ============================================================================
# -------------------------- Public plot functions ---------------------------
# ============================================================================


def polar_indmemdiff_latlon_plot(vn, var, run, unit, arr, ptype, plot_dict,
                                 title, season):
    """Single‑panel *case minus reference* plot."""
    # Projection / geometry -----------------------------------------------------------------
    proj, extent, space = _get_projection(var)
    fig, ax = plt.subplots(figsize=(10, 11), subplot_kw={"projection": proj})
    _polar_axis(ax, extent, space)

    # Data prep -----------------------------------------------------------------------------
    arr = arr.sel(season=season)
    levels, ticks, cbarticks = _levels_ticks(plot_dict, diff=True, arr=arr)
    cmap = _colormap(plot_dict.get("diff_cmap", plot_dict["cmap"]))

    lat = arr.lat
    lon_idx = list(arr.dims).index("lon")
    wrap_data, wrap_lon = add_cyclic_point(arr.values, coord=arr.lon, axis=lon_idx)

    img = ax.contourf(wrap_lon, lat, wrap_data, levels=levels, cmap=cmap,
                      transform=ccrs.PlateCarree())

    # Land mask overlay for temperature fields
    if vn == "ts-a":
        landsies = ncl_masks.LSMASK.where(ncl_masks.LSMASK == 1)
        lon_idx = landsies.dims.index("lon")
        wrap_lsm, wrap_lon_lsm = add_cyclic_point(landsies.values, coord=landsies.lon, axis=lon_idx)
        ax.contourf(wrap_lon_lsm, landsies.lat, wrap_lsm, colors="w",
                    transform=ccrs.PlateCarree(), zorder=300)
        ax.add_feature(cfeature.LAKES.with_scale("110m"), edgecolor="#b5b5b5",
                       facecolor="none", zorder=300)

    # Titles / decorations ------------------------------------------------------------------
    ax.set_title(run, fontsize=20, color="#0c80ab", y=1.07)
    fig.text(0.95, 0.77, "\u00a9 CVDP-LE", fontsize=10, color="#b5b5b5",
             weight="bold", alpha=0.75, ha="right", va="top")
    plt.suptitle(title, fontsize=26, y=0.9)

    # Colourbar -----------------------------------------------------------------------------
    _add_colorbar(fig, img, unit=unit, levels=levels, ticks=ticks,
                  cbarticks=cbarticks, ax_anchor=ax)
    return fig


# -----------------------------------------------------------------------------
# The next two functions share quite a lot of logic; to keep the public API
# intact they are thin wrappers that call the common _polar_panel_plot helper.
# -----------------------------------------------------------------------------

def _polar_panel_plot(arrs: Sequence, *, var: str, vn: str, season: str,
                      plot_dict: dict, ptype: str, title: str,
                      diff_arr=None, debug: bool = False) -> plt.Figure:
    """Back‑end worker for stacked polar panels (ind‑mem or ensemble)."""
    n_panels = len(arrs) if diff_arr is None else 4  # two runs + diff + rank
    proj, extent, space = _get_projection(var)

    fig, axs = plt.subplots(1, n_panels, figsize=(17 + 2.5 * n_panels, 15),
                            subplot_kw={"projection": proj}, sharex=True, sharey=True)
    if n_panels == 1:
        axs = [axs]

    # Decorative constant bits --------------------------------------------------------------
    for ax in axs:
        _polar_axis(ax, extent, space)

    unit = arrs[0].units

    # Pre‑compute anything expensive outside the loop
    landsies, wrap_lon_land, wrap_data_land = None, None, None
    if vn == "ts":
        landsies = ncl_masks.LSMASK.where(ncl_masks.LSMASK == 1)
        lon_idx_lsm = landsies.dims.index("lon")
        wrap_data_land, wrap_lon_land = add_cyclic_point(landsies.values,
                                                         coord=landsies.lon,
                                                         axis=lon_idx_lsm)

    # Main loop -----------------------------------------------------------------------------
    img_handles = []
    for idx, ax in enumerate(axs):
        # Choose which field goes in this pane ---------------------------------------------
        if diff_arr is not None and idx == 2:  # difference panel
            arr = diff_arr.sel(season=season)
            diff = True
            run_title = f"{arrs[0].run_name} - {arrs[1].run_name}"
        elif diff_arr is not None and idx == 3:  # rank mock‑up
            # placeholder rank panel demo
            shape = arrs[-1].shape[-2:]
            arr = af.zeros_array(*shape)
            diff = False
            run_title = "Rank"
        else:  # individual members / observations
            arr = arrs[idx if diff_arr is None else (1 if idx == 0 else 0)].sel(season=season)
            diff = False
            run_title = arr.run_name

        # Contour levels / colour map ------------------------------------------------------
        levels, ticks, cbarticks = _levels_ticks(plot_dict, diff=diff, arr=arr)
        cmap_name = plot_dict.get("diff_cmap" if diff else "cmap")
        cmap = _colormap(cmap_name)

        # Wrap longitudes and plot ---------------------------------------------------------
        lon_idx = arr.dims.index("lon")
        wrap_data, wrap_lon = add_cyclic_point(arr.values, coord=arr.lon, axis=lon_idx)
        img = ax.contourf(wrap_lon, arr.lat, wrap_data, levels=levels, cmap=cmap,
                          transform=ccrs.PlateCarree())
        img_handles.append(img)

        # Land mask overlay (SST‑style)
        if vn == "ts" and landsies is not None:
            ax.contourf(wrap_lon_land, landsies.lat, wrap_data_land, colors="w",
                        transform=ccrs.PlateCarree(), zorder=300)
            ax.add_feature(cfeature.LAKES.with_scale("110m"), edgecolor="#b5b5b5",
                           facecolor="none", zorder=300)

        # Titles & annotations -------------------------------------------------------------
        ax.set_title(run_title, fontsize=18, color="#0c80ab" if idx == 0 else "black",
                     y=1.07)
        if hasattr(arr, "yrs"):
            syr, eyr = arr.yrs[:2]
            ax.text(-0.065, 0.98, f"{syr}-{eyr}", transform=ax.transAxes,
                    fontsize=11, va="top")

    # Colourbars ---------------------------------------------------------------------------
    for ax, img in zip(axs, img_handles):
        levels, ticks, cbarticks = _levels_ticks(plot_dict, diff=False)
        _add_colorbar(fig, img, unit=unit, levels=levels, ticks=ticks,
                      cbarticks=cbarticks, ax_anchor=ax, fontsize=12)

    # Global labelling ---------------------------------------------------------------------
    fig.text(0.92, 0.7, "\u00a9 CVDP-LE", fontsize=10, color="#b5b5b5", weight="bold",
             alpha=0.75, ha="right", va="top")
    plt.suptitle(title, fontsize=24, y=0.75)
    plt.subplots_adjust(wspace=0.3)

    return fig


# Public wrappers --------------------------------------------------------------------------

def polar_indmem_latlon_plot(vn, var, arrs, plot_dict, title, ptype, season):
    """Two‑panel *case vs observation* plot (member + reference)."""
    return _polar_panel_plot(arrs, var=var, vn=vn, season=season, plot_dict=plot_dict,
                             ptype=ptype, title=title)


def polar_ensemble_plot(arrs, arr_diff, vn, var, season, ptype, plot_dict,
                         title, debug: bool = False):
    """Four‑panel plot: case, reference, difference, rank."""
    return _polar_panel_plot(arrs, diff_arr=arr_diff, var=var, vn=vn, season=season,
                             plot_dict=plot_dict, ptype=ptype, title=title, debug=debug)
