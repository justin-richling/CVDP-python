#!/usr/bin/env python3
"""
seasonal_plots.py

Creates plots for seasonal climatology metrics.
License: MIT
"""

import matplotlib.pyplot as plt

from vis import *
from visualization.vis_utils import *
import old_utils.avg_functions as af
lsmask, ncl_masks = af.land_mask()

def global_ensemble_plot(arrs, arr_diff, vn, season, ptype, plot_dict, title, debug=False):
    """
    Args
    ----
       - ptype:
          * spatialmean - global average of seasonally weighted means
          * trends - global average of seasonally weighted anomoly?? means
          * pattern - ??
    """

    #Try and format spacing based on number of cases
    #-----------------------------------------------
    # NOTE: ** this will have to change if figsize or dpi change **
    wspace=0.1
    y_title = .63
    sub_text_size = 11

    # Get variable plot info
    #-----------------------
    plot_info = plot_dict

    # get units
    #unit = plot_info["units"]
    unit = arrs[0].units

    # Set up plot
    #------------
    nrows = 1
    ncols = 4

    proj = ccrs.Robinson(central_longitude=210)
    fig_width = 15+(2.5*ncols)
    fig_height = 15
    fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(fig_width,fig_height),
                            facecolor='w', edgecolor='k', sharex=True, sharey=True,
                            subplot_kw={"projection": proj})

    img = []
    for r in range(0,ncols):
        if r == 2:

            levels = plot_info.get("diff_range",None)
            if not levels:
                diff = arrs[0]-arrs[1]
                diff_max = diff.max().item()
                diff_min = diff.min().item()
                levels = np.linspace(diff_min, diff_max, 20)
            else:
                levels = np.arange(*levels)

            # colorbar ticks
            ticks = np.arange(*plot_info.get("diff_ticks_range",levels))
            cbarticks = plot_info.get("diff_cbarticks", plot_info.get("cbarticks", None))
            if cbarticks is None:
                cbarticks = ticks

            # color map
            cmap = plot_info.get("diff_cmap",plot_info["cmap"])
            if not cmap in plt.colormaps():
                #print(f"Difference colormap {cmap} is NOT a valid matplotlib colormap. Trying to build from NCL...")
                cmap = get_NCL_colormap(cmap, extend='None')
        if r in [0,1]:
            # plot contour range
            levels = np.linspace(*plot_info["contour_levels_linspace"])
        
            # colorbar ticks
            ticks = np.arange(*plot_info["ticks_range"])

            cbarticks = plot_info.get("cbarticks", None)
            if cbarticks is None:
                cbarticks = ticks

            # color map
            cmap = plot_info["cmap"]
            if cmap not in plt.colormaps():
                #print(f"Ref/Sim colormap {cmap} is NOT a valid matplotlib colormap. Trying to build from NCL...")
                cmap = get_NCL_colormap(cmap, extend='None')

            

        # Start data gather/clean
        #------------------------

        # Rank plot
        if r == 3:
            arr = af.zeros_array(arrs[-1].shape[0], arrs[-1].shape[1])
            run = "Rank of Observations within Ensemble"
            cmap = bg_cmap
            levels = [-5,0,5,10,20,80,90,95,100,105]
            rank_levs = levels
            yrs_text = ''
            norm=PiecewiseNorm([0,5,10,20,80,90,95,100])
            unit = "%"
        else:
            if vn == "ts":
                # Set up normalization of data based off non-linear set of contour levels
                norm = mpl.colors.BoundaryNorm(ticks, amwg_cmap.N)
        # End if

        # Difference plot
        if r == 2:
            arr = arr_diff.sel(season=season)
            run = f"{arrs[0].run_name} - {arrs[1].run_name}"
            yrs_text = ''
        # End if

        # Case plots
        if r < 2:
            arr = arrs[r].sel(season=season)

            # Get run name
            #TODO: run names need to be better to get
            run = arr.run_name
            #run = f"{finarrs[r].run}"

            # Get start and end years for run
            syr = arr.yrs[0]
            eyr = arr.yrs[1]
            yrs_text = f'{syr}-{eyr}'
            if debug:
                print(yrs_text,"\n")
        # End if

        # Get wrapped data around zeroth longitude
        lat = arr.lat
        lon_idx = arr.dims.index('lon')
        wrap_data, wrap_lon = add_cyclic_point(arr.values, coord=arr.lon, axis=lon_idx)

        # Variable exceptions:
        if vn == "ts":
            landsies = ncl_masks.LSMASK.where(ncl_masks.LSMASK==1)
            lon_idx = landsies.dims.index('lon')

            # Set up data for land mask
            wrap_data_land, wrap_lon_land = add_cyclic_point(landsies.values,
                                                             coord=landsies.lon,
                                                             axis=lon_idx)
        if r < 2:
            wrap_data = clean_data(vn, wrap_data, ptype, diff=False)
        if r == 2:
            wrap_data = clean_data(vn, wrap_data, ptype, diff=True)

        # End data gather/clean
        #----------------------

        #print("wrap_data.shape",wrap_data.shape)

        # Start plot exceptions
        #----------------------
        # TODO: clean this up further?

        # Grab every other value for TS spatial mean
        # TODO: Fix this in the plot_dict!
        if (vn == "ts") and (ptype == "spatialmean") and (r in [0,1]):
        #if (vn == "ts" or (vn == "psl")) and (ptype == "spatialmean") and (r in [0,1]):
        #if (ptype == "spatialmean") and (r in [0,1]):
            ticks = plot_info["ticks"][::2]
            cbarticks = cbarticks[::2]
        if vn == "psl":
            #ticks = plot_info["ticks"][::2]
            cbarticks = cbarticks[::2]

        # Create a dictionary with arguments for contourf
        contourf_args = {
            'wrap_lon': wrap_lon,
            'lat': lat,
            'wrap_data': wrap_data,
            'levels': levels,
            'cmap': cmap,
            'transform': ccrs.PlateCarree()}

        # Onl add norm to contour dictionary if applicable
        if (r == 3) or ((r != 3) and (vn == 'ts')):
            contourf_args['norm'] = norm
        # End if

        # Extract the positional arguments and keyword arguments from the dictionary
        pos_args = [contourf_args.pop(key) for key in ['wrap_lon', 'lat', 'wrap_data']]

        # Add arguments for plot
        img.append(axs[r].contourf(*pos_args, **contourf_args))

        # Set individual plot title
        if r == 0:
            axs[r].set_title(run,loc='center',fontdict={'fontsize': 18,
                                 #'fontweight': 'bold',
                                'color': '#0c80ab',
                                })
        else:
            axs[r].set_title(run,loc='center',fontdict={'fontsize': 18,
                                 #'fontweight': 'bold',
                                #'color': '#0c80ab',
                                })
        # End if

        # Add land mask if TS
        #-------------------
        if vn == "ts":
            # Plot masked continents over TS plot to mimic SST's
            axs[r].contourf(wrap_lon_land,landsies.lat,wrap_data_land,
                            colors="w",
                            transform=ccrs.PlateCarree())
            # Plot lakes
            axs[r].add_feature(cfeature.LAKES.with_scale('110m'), #alpha=0, #facecolor=cfeature.COLORS['water'],
                                edgecolor="#b5b5b5", facecolor="none", zorder=300)
        #End if

        # End plot exceptions
        #--------------------

        # Add plot details
        #-----------------
        # Add coastlines
        axs[r].coastlines(color="#b5b5b5")

        # Add range of years to plot
        #props = dict(boxstyle='round', facecolor='grey', alpha=0.15)  # bbox features
        axs[r].text(-0.065, 0.98, yrs_text, transform=axs[r].transAxes,
                    fontsize=sub_text_size, verticalalignment='top')#, bbox=props)

        # Set up inserted colorbar axis
        axins = inset_axes(axs[r], width="85%", height="8%",
                            loc='lower center', borderpad=-3)

        # COLORBARS
        #--------------
        # Format colorbar for plots other than Rank:
        if r != 3:
            if vn == "ts":
                if ptype == "trends":
                    # Define specific tick locations for the colorbar
                    ticks = levels
                    # Create a list of labels where only the selected labels are shown
                    tick_labels = [str(loc) if loc in cbarticks else '' for loc in ticks]
                if ptype == "spatialmean":
                    # Define specific tick locations for the colorbar
                    ticks = levels
                    # Create a list of labels where only the selected labels are shown
                    tick_labels = [str(int(loc)) if loc in cbarticks else '' for loc in ticks]
            elif vn == "psl":
                if ptype == "spatialmean":
                    # Define specific tick locations for the colorbar
                    ticks = levels
                    # Create a list of labels where only the selected labels are shown
                    tick_labels = [str(int(loc)) if loc in cbarticks else '' for loc in ticks]
                    #tick_labels = [str(v) if v <= 1 else str(int(v)) for v in ticks]
                else:
                    cbarticks = ticks
                    tick_labels = [str(int(loc)) if loc in cbarticks else '' for loc in ticks]
            else:
                tick_labels = [str(int(loc)) if loc in cbarticks else '' for loc in ticks]
            #End if
        else:
            # colorbar ticks for Rank
            cbarticks = [0,5,10,20,80,90,95,100]
            ticks = rank_levs
            # Create a list of labels where only the selected labels are shown
            tick_labels = [str(loc) if loc in cbarticks else '' for loc in ticks]
        # End if

        # Set up colorbar
        #----------------
        cb = fig.colorbar(img[r], orientation='horizontal',
                        cax=axins, ticks=ticks, extend='both')

        # Format colorbar
        #----------------
        # Set the ticks on the colorbar
        cb.set_ticks(ticks)
        # 
        cb.set_ticklabels(tick_labels)
        # Set title of colorbar to units
        cb.ax.set_xlabel(unit,fontsize=18)
        # Set tick label size and remove the tick lines (optional)
        cb.ax.tick_params(labelsize=12, size=0)
        # Remove border of colorbar
        cb.outline.set_visible(False)

    madeup_r = 0.28
    r_text = f'r={madeup_r}'
    axs[0].text(.875, 0.98, r_text, transform=axs[0].transAxes, fontsize=sub_text_size, verticalalignment='top')
    axs[-1].text(.875, 0.99, "--%", transform=axs[-1].transAxes, fontsize=12, verticalalignment='top')

    fig.text(0.92, 0.61, "$\\copyright$ CVDP-LE", fontsize=10, color='#b5b5b5', weight='bold', alpha=0.75, ha='right', va='top')

    # Set figure title
    plt.suptitle(title, fontsize=24, y=y_title)

    # Clean up the spacing a bit
    plt.subplots_adjust(wspace=wspace)

    return fig