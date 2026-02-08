#!/usr/bin/env python3
"""
polar_plots.py

Creates plots for seasonal polar climatology metrics.
License: MIT
"""

import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cartopy.util import add_cyclic_point
import cartopy.feature as cfeature
import matplotlib.path as mpath

from vis import *
from vis.vis_utils import *
import cvdp_utils.avg_functions as af
lsmask, ncl_masks = af.land_mask()
import cvdp_utils.analysis as an


#def polar_indmemdiff_latlon_plot(vn, var, runs, arr, ptype, plot_dict, title):
def polar_indmemdiff_latlon_plot(vn, var, arrs, plot_dict, title, ptype):
    '''
    Docstring for polar_indmemdiff_latlon_plot
    
    :param vn: Description
    :param arrs: Description
    :param plot_dict: Description
    :param title: Description
    :param ptype: Description

    arrs is now a list of lists!
        first entry is list of simulations
        second entry is list of references
    ''' 
    nh_vars = ["NAM","psl"]
    sh_vars = ["SAM", "PSA1", "PSA2","psl"]

    y_title = .715

    # Get variable plot info
    # -----------------------
    plot_info = plot_dict

    # plot contour range
    levels = None
    if "diff_levels_linspace" in plot_info:
        #print('plot_info["diff_levels_linspace"]',plot_info["diff_levels_linspace"])
        levels = np.linspace(*plot_info["diff_levels_linspace"])
    if "diff_levels_range" in plot_info:
        #print('plot_info["diff_levels_range"]',plot_info["diff_levels_range"])
        levels = np.arange(*plot_info["diff_levels_range"])
    if "diff_levels_list" in plot_info:
        #print('plot_info["diff_levels_list"]',plot_info["diff_levels_list"])
        levels = np.array(plot_info["diff_levels_list"])
        good_list = True
    #print("type(levels)",type(levels))
    if not isinstance(levels,np.ndarray) and not good_list:
        diff_max = arr.max().item()
        diff_min = arr.min().item()
        levels = np.linspace(diff_min, diff_max, 20)

    cbarticks = plot_info.get("diff_cbar_labels", levels)
    # colorbar ticks

    # color map
    cmap = plot_info.get("diff_cmap","PuOr")
    if cmap not in plt.colormaps():
        cmap = get_NCL_colormap(cmap, extend='None')
    # get units
    if isinstance(arrs[0][0].units, str):
        unit = arrs[0][0].units
    else:
        unit = arrs[0][0].units.values
    

    # Set up figure and axes
    if var in nh_vars:
        proj = ccrs.NorthPolarStereo(central_longitude=0)
        extent = [-180, 180, 20, 90]
        space = 16.5
    if var in sh_vars:
        proj = ccrs.SouthPolarStereo(central_longitude=0)
        extent = [-180, 180, -20, -90]
        space = -16.5

    #fig_width = 10
    #fig_height = 10+(3) #try and dynamically create size of fig based off number of cases (therefore rows)
    #fig, axs = plt.subplots(nrows=1,ncols=1,figsize=(fig_width,fig_height), facecolor='w', edgecolor='k',
    #                        sharex=True,
    #                        sharey=True,
    #                        subplot_kw={"projection": proj})
    

    # Create subplots
    n_cases = len(arrs)
    ncols = 10
    nrows = (n_cases + ncols - 1) // ncols  # Calculate the required rows
    if n_cases <= ncols:
        ncols = n_cases
    #print("n_cases",n_cases,"nrows",nrows,"ncols",ncols)

    #proj = WinkelTripel(central_longitude=210)
    if n_cases == 2 or n_cases == 3 or n_cases == 4:
        hgt = nrows*2
        wdth = ncols*3
    else:
        hgt = nrows*2.5
        wdth = ncols*4
    hgt = nrows*2.5
    fig, axs = plt.subplots(nrows, ncols, figsize=(wdth, hgt),
                             facecolor="w", edgecolor="k", sharex=True, sharey=True,
                             subplot_kw={"projection": proj},constrained_layout=False,squeeze=False)

    #if n_cases > 10:
    #    axs = axs.flatten()
    axs = axs.ravel()

    # Set empty list for contour plot objects
    img = []
    #print(type(arrs[0]),arrs[0])
    for i,arr in enumerate(arrs):
        # Grab run metadata for plots
        # ----------------------------
        # Data years for this run
        #syr = arr.yrs[0]
        #eyr = arr.yrs[-1]

        # Run name
        #run = f"{arr.run}"

        # For having 180 as the cental longitude (Pacific centric view), sometimes the data and longitude
        # have to be "wrapped" around this lingitude. Is this an xarray problem?
        # NOTE: Maybe not necessary anymore
        lon_idx = arr.dims.index("lon")
        wrap_data, wrap_lon = add_cyclic_point(
                arr.values, coord=arr.lon, axis=lon_idx
        )
        lat = arr.lat

        axs[i].set_extent(extent, ccrs.PlateCarree())
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        axs[i].set_boundary(circle, transform=axs[i].transAxes)

        # Create a dictionary with arguments for contourf
        contourf_args = {
                "wrap_lon": wrap_lon,
                "lat": lat,
                "levels": levels,
                "cmap": cmap,
                "transform": ccrs.PlateCarree(),
            }
        wrap_data = clean_data(vn, wrap_data, ptype, diff=False)

        if vn == "ts":
            # Land mask
            # ----------
            # Mask out land using masking data
            land_data = ncl_masks.LSMASK.where(ncl_masks.LSMASK == 1)

            # Set up data for land mask
            lon_idx = land_data.dims.index("lon")
            wrap_data_land, wrap_lon_land = add_cyclic_point(
                    land_data.values, coord=land_data.lon, axis=lon_idx
                )

            # Set up normalization of data based off non-linear set of contour levels
            wrap_data = np.where(wrap_data < 0, 0, wrap_data)
            norm = mpl.colors.BoundaryNorm(levels, amwg_cmap.N)
            contourf_args["norm"] = norm
            # Plot masked continents over TS plot to mimic SST's
            axs[i].contourf(wrap_lon_land, land_data.lat, wrap_data_land, colors="w",
                                transform=ccrs.PlateCarree(), zorder=300)
            # Plot lakes
            axs[i].add_feature(cfeature.LAKES.with_scale("110m"),
                               edgecolor="#b5b5b5", facecolor="none", zorder=300)

        if vn == "psl":
            wrap_data = np.where(wrap_data < -9, -9, wrap_data)

        contourf_args["wrap_data"] = wrap_data
        # Extract the positional arguments and keyword arguments from the dictionary
        pos_args = [contourf_args.pop(key) for key in ["wrap_lon", "lat", "wrap_data"]]

        # Create a filled contour plot using the dictionary of arguments
        img.append(axs[i].contourf(*pos_args, **contourf_args))

        # Define longitude labels
        lon_ticks = np.arange(0, 360, 30)
        lon_labels = ['0', '30E', '60E', '90E', '120E', '150E', '180', '150W', '120W', '90W', '60W', '30W']
        #lon_labels = ['0°', '30°E', '60°E', '90°E', '120°E', '150°E', '180°', '150°W', '120°W', '90°W', '60°W', '30°W']
        
        for lon, label in zip(lon_ticks, lon_labels):
            x, y = ccrs.PlateCarree().transform_point(lon, space, ccrs.PlateCarree())
            axs[i].text(x, y, label, transform=ccrs.PlateCarree(), fontsize=14, ha='center', va='center')

        plt.setp(axs[i].spines.values(), lw=.5, color='grey', alpha=0.7)

        # Add coast lines and title
        axs[i].coastlines("50m", color="#b5b5b5")
        #axs[i].set_title(run, loc='center', fontdict={'fontsize': 20, 'color': '#0c80ab'}, y=1.07)
        # Add r value to case run plot
        # TODO: Calculate r-values
        if i == 0:
            madeup_r = 0.98
            r_text = f"r={madeup_r}"
            axs[i].text(0.93, 0.98, r_text, transform=axs[i].transAxes, fontsize=10, verticalalignment="top",)
        # End if

    # COLORBARS
    # ----------------
    # Set up axis to insert into color bar
    #axins = inset_axes(axs[-1], width="100%", height="5%", loc="lower center", borderpad=-5)

    # Format the colorbar depending on the plot type and variable
    #FLAG: cleaned this up

    #axins = inset_axes(axs, width="120%", height="5%", loc='lower center', borderpad=-9)
    #tick_font_size = 16

    if vn == "ts":
        if ptype == "trends":
            # Define specific tick locations for the colorbar
            ticks = levels
            # Create a list of labels where only the selected labels are shown
            tick_labels = [str(loc) if loc in cbarticks else '' for loc in ticks]
        if ptype == "spatialmean":
            # Define the locations for custom set of labels
            #cbarticks = np.arange(-5,6,1)

            # Define specific tick locations for the colorbar
            ticks = cbarticks
            # Create a list of labels where only the selected labels are shown
            tick_labels = [str(int(loc)) if loc in cbarticks else '' for loc in ticks]
    else:
        ticks = cbarticks
        tick_labels = [str(int(loc)) if loc in cbarticks else '' for loc in ticks]

    # Set up colorbar
    #----------------
    # Add colorbar under last row (partial row handled)
    #cb = fig.colorbar(img, orientation='horizontal',
    #                    cax=axins, ticks=ticks, extend='both')
    #                    #cax=axins, ticks=tick_labels, extend='both')
    cbar = add_centered_colorbar(fig, axs, img[0], unit, ticks,
                          n_cols_per_row=10,
                          pad_inches=0.75,
                          height_inches=0.35)

    # Turn off unused axes
    for j in range(n_cases, len(axs)):
        axs[j].axis("off")


    # Set values to floats for decimals and int for integers for tick labels
    #bound_labels = [str(v) if v <= 1 else str(int(v)) for v in ticks]
    #cb.set_ticklabels(bound_labels, size=0)

    """# Format colorbar
    #----------------        
    # Set tick label size and remove the tick lines (optional)
    cb.ax.tick_params(labelsize=12, size=0)
    # Remove border of colorbar
    #cb.outline.set_visible(False)
    cb.outline.set_edgecolor("grey")
    cb.outline.set_linewidth(0.6)

    # Add CVDP watermark
    fig.text(0.95, 0.77, "$\\copyright$ CVDP-LE", fontsize=10, color='#b5b5b5', weight='bold', 
             alpha=0.75, ha='right', va='top')

    #Set figure title
    plt.suptitle(title, fontsize=26, y=0.9)"""

    fig.text(0.9, 0.82, "$\\copyright$ CVDP-LE", fontsize=10, color='#b5b5b5', weight='bold', alpha=0.75, ha='right', va='top')
    #title = f"{title} constrained_layout=true hspace=0.05, ytitle=0.9, y-height=nrows*4"
    #title = f"{title} constrained_layout=true, hspace=0.05, ytitle=0.99, y-height=nrows*2.5"
    if n_cases == 2 or n_cases == 3 or n_cases == 4:
        fontsize = 20
        y_title = 0.99
    else:
        fontsize = 26
    plt.suptitle(title, fontsize=fontsize, y=y_title, x=0.515)  # y=0.325 y=0.225

    # Clean up the spacing a bit
    """if n_cases == 2 or n_cases == 3 or n_cases == 4:
        hspace = -0.03
    else:
        hspace = 0.05"""
    #hspace = 0.05
    #plt.subplots_adjust(hspace=hspace)

    if n_cases == 2 or n_cases == 3 or n_cases == 4:
        plt.subplots_adjust(
            top=0.70,     # lower this → MORE space between title and plots
            bottom=0.15   # raise this → LESS space between plots and colorbar
        )
    else:
        hspace = 0.05
        plt.subplots_adjust(hspace=hspace)
    hspace = 0.05
    plt.subplots_adjust(hspace=hspace,wspace=0.03)
    return fig

    return fig