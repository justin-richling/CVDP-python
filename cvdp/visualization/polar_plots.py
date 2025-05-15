#!/usr/bin/env python3
"""
polar_plots.py

Creates plots for seasonal polar climatology metrics.
License: MIT
"""

import matplotlib.pyplot as plt

from vis import *
from visualization.vis_utils import *
import old_utils.avg_functions as af
lsmask, ncl_masks = af.land_mask()


def polar_diff_plot(vn, var, run, arr, ptype, plot_dict, title, plot_name, debug=False):
    nh_vars = ["NAM"]
    sh_vars = ["SAM", "PSA1", "PSA2"]

    y_title = .715

    # Get variable plot info
    #-----------------------
    plot_info = plot_dict[vn]

   # plot contour range
    #levels = plot_info["range"]
    levels = plot_info.get("diff_range",plot_info["range"])

    # colorbar ticks
    #ticks = plot_info["ticks"]
    ticks = plot_info.get("diff_ticks",plot_info["ticks"])

    cbarticks = plot_info.get("diff_cbarticks", plot_info.get("cbarticks", None))
    #plot_info.get("diff_cbarticks", None)
    if cbarticks is None:
        cbarticks = ticks

    # color map
    cmap = plot_info["cmap"]

    # get units
    unit = plot_info["units"]

    # Set up figure and axes
    if var in nh_vars:
        proj = projection=ccrs.NorthPolarStereo(central_longitude=0)
        extent = [-180, 180, 20, 90]
        space = 16.5
    if var in sh_vars:
        proj = projection=ccrs.SouthPolarStereo(central_longitude=0)
        extent = [-180, 180, -20, -90]
        space = -16.5

    fig_width = 10
    fig_height = 10+(3) #try and dynamically create size of fig based off number of cases (therefore rows)
    fig, axs = plt.subplots(nrows=1,ncols=1,figsize=(fig_width,fig_height), facecolor='w', edgecolor='k',
                            sharex=True,
                            sharey=True,
                            subplot_kw={"projection": proj})

    # Get wrapped data around zeroth longitude
    lat = arr.lat
    #lon_idx = arr.dims.index('lon')
    dimension_list = list(arr.dims.keys())  # Convert dimensions to a list
    lon_idx = dimension_list.index('lon') 
    print("lon_idx",lon_idx,"\n")
    wrap_data, wrap_lon = add_cyclic_point(arr.values, coord=arr.lon, axis=lon_idx)


    # End data gather/clean
    #----------------------

    #for r in range(nrows):
    if 1==1:
        axs.set_extent(extent, ccrs.PlateCarree())
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        axs.set_boundary(circle, transform=axs.transAxes)

        # Data years
        #syr = finarrs[vn].time[0].dt.year.values
        #eyr = finarrs[vn].time[-1].dt.year.values

        # Run name
        #run = finarrs[vn].run

        #arr = arrs

        #lon_idx = arr.dims.index('lon')
        wrap_data, wrap_lon = add_cyclic_point(arr.values, coord=arr.lon, axis=lon_idx)
        lat = arr.lat

        # Create a dictionary with arguments for contourf
        contourf_args = {
            'wrap_lon': wrap_lon,
            'lat': lat,
            'levels': levels,
            'cmap': cmap,
            'transform': ccrs.PlateCarree()
        }

        if vn == "ts-a":
            landsies = ncl_masks.LSMASK.where(ncl_masks.LSMASK == 1)
            lon_idx = landsies.dims.index('lon')
            wrap_data_land, wrap_lon_land = add_cyclic_point(landsies.values, coord=landsies.lon, axis=lon_idx)
            wrap_data = np.where(wrap_data < 0, 0, wrap_data)
            norm = mpl.colors.BoundaryNorm(levels, amwg_cmap.N)
            contourf_args['norm'] = norm
            axs[r].contourf(wrap_lon_land, landsies.lat, wrap_data_land, colors="w", transform=ccrs.PlateCarree(), zorder=300)
            axs[r].add_feature(cfeature.LAKES.with_scale('110m'), edgecolor="#b5b5b5", facecolor="none", zorder=300)

        if vn == "psl-a":
            wrap_data = np.where(wrap_data < -9, -9, wrap_data)

        contourf_args['wrap_data'] = wrap_data
        pos_args = [contourf_args.pop(key) for key in ['wrap_lon', 'lat', 'wrap_data']]
        img = axs.contourf(*pos_args, **contourf_args)

        # Define longitude labels
        lon_ticks = np.arange(0, 360, 30)
        lon_labels = ['0', '30E', '60E', '90E', '120E', '150E', '180', '150W', '120W', '90W', '60W', '30W']
        #lon_labels = ['0°', '30°E', '60°E', '90°E', '120°E', '150°E', '180°', '150°W', '120°W', '90°W', '60°W', '30°W']
        
        for lon, label in zip(lon_ticks, lon_labels):
            x, y = ccrs.PlateCarree().transform_point(lon, space, ccrs.PlateCarree())
            axs.text(x, y, label, transform=ccrs.PlateCarree(), fontsize=14, ha='center', va='center')

        plt.setp(axs.spines.values(), lw=.5, color='grey', alpha=0.7)

        axs.coastlines('50m', color="#b5b5b5", alpha=0.5)
        axs.set_title(run, loc='center', fontdict={'fontsize': 20, 'color': '#0c80ab'}, y=1.07)

        #yrs_text = f'{syr}-{eyr}'
        #axs.text(0.0, 0.93, yrs_text, transform=axs.transAxes, fontsize=12, verticalalignment='top')

        """
        madeup_percent = "23.6%"
        r_text = madeup_percent
        axs.text(.85, 0.93, r_text, transform=axs.transAxes, fontsize=12, verticalalignment='top')

        if r == 0:
            madeup_r = f"r={0.77}"
            r_text = madeup_r
            axs.text(.85, 0.9, r_text, transform=axs.transAxes, fontsize=12, verticalalignment='top')
        """

    axins = inset_axes(axs, width="120%", height="5%", loc='lower center', borderpad=-9)
    tick_font_size = 16

    if vn == "ts":
        if ptype == "trends":
            # Define the locations for custom set of labels
            #cbarticks = [-6, -4, -2, -0.5, 0, 0.5, 2, 4, 6]
            #print("ts trends cbarticks",cbarticks)

            # Define specific tick locations for the colorbar
            ticks = levels
            # Create a list of labels where only the selected labels are shown
            tick_labels = [str(loc) if loc in cbarticks else '' for loc in ticks]
        if ptype == "spatialmean":
            # Define the locations for custom set of labels
            #cbarticks = np.arange(-5,6,1)
            #print("ts spatialmean cbarticks",cbarticks)

            # Define specific tick locations for the colorbar
            ticks = levels
            # Create a list of labels where only the selected labels are shown
            tick_labels = [str(int(loc)) if loc in cbarticks else '' for loc in ticks]
    else:
        #cbarticks = ticks
        #print("cbarticks",cbarticks)
        tick_labels = [str(int(loc)) if loc in cbarticks else '' for loc in ticks]
    #End if

    # Set up colorbar
    #----------------
    cb = fig.colorbar(img, orientation='horizontal',
                     cax=axins,
                     ticks=ticks
                     )
    # Set tick label font size
    tick_font_size = 16

    # Format colorbar
    #----------------
    # Set the ticks on the colorbar
    cb.set_ticks(ticks)
        
    # 
    cb.set_ticklabels(tick_labels)

    # Set title of colorbar to units
    cb.ax.set_xlabel(unit,fontsize=18)

    # Set tick label size and remove the tick lines (optional)
    cb.ax.tick_params(labelsize=16, size=0)

    # Remove border of colorbar
    cb.outline.set_visible(False)

    # Add CVDP watermark
    fig.text(0.95, 0.77, "$\\copyright$ CVDP-LE", fontsize=10, color='#b5b5b5', weight='bold', 
             alpha=0.75, ha='right', va='top')

    #Set figure title
    plt.suptitle(title, fontsize=26, y=0.9)

    # Save figure
    #------------
    plt.savefig(plot_name,bbox_inches="tight")

    # If debugging, go ahead and show the plot
    if debug:
        plt.show()
    else:
        plt.close()









def stacked_polar_plot(vn, var, finarrs, arrs, plot_dict, title, plot_name, ptype, season, debug=False):
    nrows = 2
    ncols = 1

    # Format spacing
    hspace = 0.6
    y_title = .79

    # Get variable plot info
    plot_info = plot_dict[var]

    # Plot contour range
    levels = plot_info["range"]

    # Colorbar ticks
    ticks = plot_info["ticks"]

    # Color map
    cmap = plot_info["cmap"]

    # Units
    unit = plot_info["units"]

    # Define projection and extent
    if var == "NAM" or var == "PNO" or var == "PNA":
        proj = projection=ccrs.NorthPolarStereo(central_longitude=0)
        extent = [-180, 180, 20, 90]
        space = 17
    if var == "SAM" or var == "PSA1" or var == "PSA2":
        proj = projection=ccrs.SouthPolarStereo(central_longitude=0)
        extent = [-180, 180, -20, -90]
        space = -17

    # Set up plot
    fig_width = 6
    fig_height = 22
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height), facecolor='w', edgecolor='k',
                            sharex=True, sharey=True, subplot_kw={"projection": proj})

    #img = []
    #for r in range(nrows):
     # Set empty list for contour plot objects
    img = []
    for i,r in enumerate([1,0]): # Plot obs first (second array in list) then case (first array in list)
        axs[i].set_extent(extent, ccrs.PlateCarree())
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        axs[i].set_boundary(circle, transform=axs[i].transAxes)

        # Data years
        syr = finarrs[r].time[0].dt.year.values
        eyr = finarrs[r].time[-1].dt.year.values

        # Run name
        run = finarrs[r][vn].run

        arr = arrs[r]

        lon_idx = arr.dims.index('lon')
        wrap_data, wrap_lon = add_cyclic_point(arr.values, coord=arr.lon, axis=lon_idx)
        lat = arr.lat

        # Create a dictionary with arguments for contourf
        contourf_args = {
            'wrap_lon': wrap_lon,
            'lat': lat,
            'levels': levels,
            'cmap': cmap,
            'transform': ccrs.PlateCarree()
        }

        contourf_args['wrap_data'] = wrap_data
        pos_args = [contourf_args.pop(key) for key in ['wrap_lon', 'lat', 'wrap_data']]
        img.append(axs[i].contourf(*pos_args, **contourf_args))

        # Define longitude labels
        lon_ticks = np.arange(0, 360, 30)
        lon_labels = ['0', '30E', '60E', '90E', '120E', '150E', '180', '150W', '120W', '90W', '60W', '30W']
        #lon_labels = ['0°', '30°E', '60°E', '90°E', '120°E', '150°E', '180°', '150°W', '120°W', '90°W', '60°W', '30°W']
        
        for lon, label in zip(lon_ticks, lon_labels):
            x, y = ccrs.PlateCarree().transform_point(lon, space, ccrs.PlateCarree())
            axs[i].text(x, y, label, transform=ccrs.PlateCarree(), fontsize=9, ha='center', va='center')

        plt.setp(axs[i].spines.values(), lw=.5, color='grey', alpha=0.7)

        axs[i].coastlines('50m', color="#b5b5b5", alpha=0.5)
        axs[i].set_title(run, loc='center', fontdict={'fontsize': 16, 'color': '#0c80ab'}, y=1.07)

        yrs_text = f'{syr}-{eyr}'
        axs[i].text(0.0, 0.93, yrs_text, transform=axs[i].transAxes, fontsize=12, verticalalignment='top')

        madeup_percent = "23.6%"
        r_text = madeup_percent
        axs[i].text(.85, 0.93, r_text, transform=axs[i].transAxes, fontsize=12, verticalalignment='top')

        if r == 0:
            madeup_r = f"r={0.77}"
            r_text = madeup_r
            axs[i].text(.85, 0.9, r_text, transform=axs[i].transAxes, fontsize=12, verticalalignment='top')

    fig.text(0.95, 0.77, "$\\copyright$ CVDP-LE", fontsize=10, color='#b5b5b5', weight='bold', alpha=0.75, ha='right', va='top')

    axins = inset_axes(axs[-1], width="120%", height="5%", loc='lower center', borderpad=-5)
    tick_font_size = 16

     # Format the colorbar depending on the plot type and variable
    #FLAG: cleaned this up
    if vn == "ts":
        if ptype == "trends":
            # Define the locations for custom set of labels
            #cbarticks = [-6, -4, -2, -0.5, 0, 0.5, 2, 4, 6]

            # Define specific tick locations for the colorbar
            ticks = levels
            # Create a list of labels where only the selected labels are shown
            tick_labels = [str(loc) if loc in cbarticks else '' for loc in ticks]
        if ptype == "spatialmean":
            # Define the locations for custom set of labels
            #cbarticks = np.arange(0,37,2)

            # Define specific tick locations for the colorbar
            ticks = levels
            # Create a list of labels where only the selected labels are shown
            tick_labels = [str(int(loc)) if loc in cbarticks else '' for loc in ticks]
    else:
        cbarticks = ticks
        tick_labels = [str(int(loc)) if loc in cbarticks else '' for loc in ticks]
    #print("ticks:",ticks)
    #print("tick_labels:",tick_labels)

    # Set up colorbar
    #----------------
    cb = fig.colorbar(img[-1], orientation="horizontal", cax=axins, ticks=ticks)

    # Format colorbar
    #----------------
    # Set the ticks on the colorbar
    cb.set_ticks(ticks)
        
    # Set tick label
    cb.set_ticklabels(tick_labels)

    # Set title of colorbar to units
    cb.ax.set_xlabel(unit,fontsize=18)

    # Set tick label size and remove the tick lines (optional)
    cb.ax.tick_params(labelsize=16, size=0)

    # Remove border of colorbar
    cb.outline.set_visible(False)

    plt.suptitle(title, fontsize=22, y=y_title)
    plt.subplots_adjust(hspace=-0.475)
    plt.savefig(plot_name, bbox_inches="tight")

    # If debugging, go ahead and show the plot
    if debug:
        plt.show()
    else:
        plt.close()



def polar_ensemble_plot(finarrs, arrs, arr_diff, vn, var, season, ptype, plot_dict, title, plot_name, debug=False):
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
    wspace=0.3
    y_title = .72
    sub_text_size = 11

    nh_vars = ["NAM"]
    sh_vars = ["SAM", "PSA1", "PSA2"]
    eof_vars = nh_vars+sh_vars

    # Get variable plot info
    #-----------------------
    plot_info = plot_dict[vn]

    # get units
    unit = plot_info["units"]

    # Set up figure and axes
    if var in nh_vars:
        proj = projection=ccrs.NorthPolarStereo(central_longitude=0)
        extent = [-180, 180, 20, 90]
        space = 14 #16.5
    if var in sh_vars:
        proj = projection=ccrs.SouthPolarStereo(central_longitude=0)
        extent = [-180, 180, -20, -90]
        space = -14 #-16.5

    # Set up plot
    #------------
    nrows = 1
    ncols = 4


    fig_width = 17+(2.5*ncols)
    fig_height = 15

    fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(fig_width,fig_height), facecolor='w', edgecolor='k',
                            sharex=True,
                            sharey=True,
                            subplot_kw={"projection": proj})


    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    # Define longitude labels
    lon_ticks = np.arange(0, 360, 30)
    lon_labels = ['0', '30E', '60E', '90E', '120E', '150E', '180', '150W', '120W', '90W', '60W', '30W']


    img = []
    for r in range(0,ncols):
        axs[r].set_extent(extent, ccrs.PlateCarree())
        axs[r].set_boundary(circle, transform=axs[r].transAxes)


        for lon, label in zip(lon_ticks, lon_labels):
            x, y = ccrs.PlateCarree().transform_point(lon, space, ccrs.PlateCarree())
            axs[r].text(x, y, label, transform=ccrs.PlateCarree(), fontsize=14, ha='center', va='center')

        plt.setp(axs[r].spines.values(), lw=.5, color='grey', alpha=0.7)

        axs[r].coastlines('50m', color="#b5b5b5", alpha=0.5)

        if r == 2:
            levels = plot_info.get("diff_range",plot_info["range"])

            # colorbar ticks
            ticks = plot_info.get("diff_ticks",plot_info["ticks"])

            #cbarticks = plot_info.get("diff_cbarticks", None)
            cbarticks = plot_info.get("diff_cbarticks", plot_info.get("cbarticks", None))
            if cbarticks is None:
                cbarticks = ticks

            # color map
            cmap = plot_info.get("diff_cmap",plot_info["cmap"])

            arr = arr_diff
            run = f"{finarrs[0].run} - {finarrs[1].run}"
            yrs_text = ''
        if r in [0,1]:
            # plot contour range
            levels = plot_info["range"]
        
            # colorbar ticks
            ticks = plot_info["ticks"]

            cbarticks = plot_info.get("cbarticks", None)
            if cbarticks is None:
                cbarticks = ticks

            # color map
            cmap = plot_info["cmap"]

            arr = arrs[r]

            # Get run name
            #TODO: run names need to be better to get
            run = f"{finarrs[r].run}"

            # Get start and end years for run
            syr = finarrs[r].yrs[0]
            eyr = finarrs[r].yrs[1]
            yrs_text = f'{syr}-{eyr}'
            if debug:
                print(yrs_text,"\n")

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
            norm = PiecewiseNorm([0,5,10,20,80,90,95,100])
            unit = "%"
        else:
            if vn == "ts":
                # Set up normalization of data based off non-linear set of contour levels
                norm = mpl.colors.BoundaryNorm(ticks, amwg_cmap.N)
        # End if
        '''
        # Difference plot
        if r == 2:
            arr = arr_diff
            run = f"{finarrs[0].run} - {finarrs[1].run}"
            yrs_text = ''
        # End if
        '''
        '''
        # Case plots
        if r < 2:
            arr = arrs[r]

            # Get run name
            #TODO: run names need to be better to get
            run = f"{finarrs[r].run}"

            # Get start and end years for run
            syr = finarrs[r].yrs[0]
            eyr = finarrs[r].yrs[1]
            yrs_text = f'{syr}-{eyr}'
            if debug:
                print(yrs_text,"\n")
        # End if
        '''

        # Get wrapped data around zeroth longitude
        lat = arr.lat
        lon_idx = arr.dims.index('lon')
        wrap_data, wrap_lon = add_cyclic_point(arr.values, coord=arr.lon, axis=lon_idx)

        axs[r].set_title(run, loc='center', fontdict={'fontsize': 20, 'color': '#0c80ab'}, y=1.07)

        # Variable exceptions:
        if vn == "ts":

            landsies = ncl_masks.LSMASK.where(ncl_masks.LSMASK==1)
            lon_idx = landsies.dims.index('lon')

            # Set up data for land mask
            wrap_data_land, wrap_lon_land = add_cyclic_point(landsies.values,
                                                             coord=landsies.lon,
                                                             axis=lon_idx)

        # Clean the data
        if r < 2:
            wrap_data = clean_data(vn, wrap_data, ptype, diff=False)
        if r == 2:
            wrap_data = clean_data(vn, wrap_data, ptype, diff=True)

        # End data gather/clean
        #----------------------


        # Start plot exceptions
        #----------------------
        # TODO: clean this up further?

        # Grab every other value for TS spatial mean
        # TODO: Fix this in the plot_dict!
        if (vn == "ts") and (ptype == "spatialmean") and (r in [0,1]):
            ticks = plot_info["ticks"][::2]
            cbarticks = cbarticks[::2]
        if vn == "psl":
            ticks = plot_info["ticks"][::2]
            cbarticks = cbarticks[::2]

        # Create a dictionary with arguments for contourf
        contourf_args = {'wrap_lon': wrap_lon, 'lat': lat,
                        'wrap_data': wrap_data, 'levels': levels,
                        'cmap': cmap, 'transform': ccrs.PlateCarree()}

        # Onl add norm to contour dictionary if applicable
        if (r == 3) or ((r != 3) and (vn == 'ts')):
            contourf_args['norm'] = norm
        # End if

        # Extract the positional arguments and keyword arguments from the dictionary
        pos_args = [contourf_args.pop(key) for key in ['wrap_lon', 'lat', 'wrap_data']]

        # Add arguments for plot
        img.append(axs[r].contourf(*pos_args, **contourf_args))

        # Set individual plot title
        y_sub = 1.07
        if r == 0:
            axs[r].set_title(run,loc='center',fontdict={'fontsize': 18,
                                 #'fontweight': 'bold',
                                'color': '#0c80ab',
                                },y=y_sub)
        else:
            axs[r].set_title(run,loc='center',fontdict={'fontsize': 18,
                                 #'fontweight': 'bold',
                                #'color': '#0c80ab',
                                },y=y_sub)
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
                            loc='lower center', borderpad=-5)

        # COLORBARS
        #--------------
        #cbarticks = ticks
        #print("cbarticks ensemble",r,ptype,cbarticks,"\n")
        
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
                #print("tick_labels ensemble",r,ptype,tick_labels,"\n")
            else:
                #cbarticks = ticks
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

    fig.text(0.92, 0.7, "$\\copyright$ CVDP-LE", fontsize=10, color='#b5b5b5', weight='bold', alpha=0.75, ha='right', va='top')

    # Set figure title
    plt.suptitle(title, fontsize=24, y=y_title)

    # Clean up the spacing a bit
    plt.subplots_adjust(wspace=wspace)

    # Save figure
    #------------
    plt.savefig(plot_name,bbox_inches="tight")

    # If debugging, go ahead and show the plot
    if debug:
        plt.show()
    else:
        plt.close()
