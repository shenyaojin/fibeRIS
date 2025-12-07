# Enhanced viz tools for multi-stage hydraulic fracturing
# Shenyao Jin, 02/2025

import os
import matplotlib.pyplot as plt
import numpy as np
from fiberis.io.reader_moose_vpp import MOOSEVectorPostProcessorReader
from fiberis.io.reader_moose_ps import MOOSEPointSamplerReader
from mpl_toolkits.axes_grid1 import make_axes_locatable
from fiberis.analyzer.Data1D.core1D import Data1D
from fiberis.analyzer.Data2D.core2D import Data2D


# ------ MOOSE related plotting functions ------

# MOOSE post processor plotting function: point sampler
def plot_point_samplers(folder, output_dir):
    """
    Reads all point sampler data from a given folder and plots each variable.

    Args:
        folder (str): The folder containing the point sampler data.
        output_dir (str): The directory to save the output plots.
    """
    point_samplers = MOOSEPointSamplerReader()
    # Get the max variable index
    max_ind = point_samplers.get_max_index(folder=folder)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(1, max_ind + 1):
        point_samplers.read(folder=folder, variable_index=i)
        plt.figure(figsize=[6, 2])
        plt.plot(point_samplers.taxis[1:], point_samplers.data[1:], label=point_samplers.variable_name)

        plt.xlabel("Time [s]")
        plt.ylabel(point_samplers.variable_name)
        plt.title(f"Point Sampler Data - Variable {i}")

        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{point_samplers.variable_name}.png")
        plt.close()

# MOOSE post processor plotting function: vector sampler
def plot_vector_samplers(folder, output_dir):
    """
    Reads all vector sampler data from a given folder and plots each variable
    for each post-processor as a pcolormesh plot.

    Args:
        folder (str): The folder containing the vector sampler data.
        output_dir (str): The directory to save the output plots.
    """
    vector_sampler = MOOSEVectorPostProcessorReader()
    max_processor_id, max_variable_index = vector_sampler.get_max_indices(folder)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(max_processor_id + 1):
        for j in range(1, max_variable_index + 1):
            vector_sampler.read(folder, i, j)

            # Calculate the distance
            distance = np.sqrt(
                (vector_sampler.xaxis - vector_sampler.xaxis[0]) ** 2 +
                (vector_sampler.yaxis - vector_sampler.yaxis[0]) ** 2
            )

            # Check if data is not empty and has more than one time step
            if vector_sampler.data is None or vector_sampler.data.shape[1] <= 1:
                print(f"Skipping plot for processor {i}, variable {j} due to insufficient data.")
                continue

            clim = (
                np.nanmin(vector_sampler.data[:, 1:]),
                np.nanmax(vector_sampler.data[:, 1:])
            )

            # Plot the data in pcolormesh
            plt.figure(figsize=(10, 6))
            plt.pcolormesh(
                vector_sampler.taxis[1:],
                distance,
                vector_sampler.data[:, 1:],
                cmap='bwr',
                clim=clim,
                shading='auto'
            )
            plt.xlabel('Time (s)')
            plt.ylabel('Distance (m)')
            plt.title(f"{vector_sampler.sampler_name} - {vector_sampler.variable_name}")
            
            # Add colorbar
            cbar = plt.colorbar()
            cbar.set_label(vector_sampler.variable_name)

            plt.tight_layout()
            
            # Save the figure
            output_filename = f"{output_dir}/{vector_sampler.sampler_name}_var_{j}.png"
            plt.savefig(output_filename)
            plt.close()


def plot_dss_and_gauge_co_plot(data2d: Data2D, data1d: Data1D,
                                    d2_plot_args: dict = None, d1_plot_args: dict = None,
                                    figsize=(7, 6)):
    """
    Plots a 2D dataset on top and a 1D dataset on the bottom, sharing the x-axis,
    with built-in sensible defaults.
    The 1D data is cropped to the time range of the 2D data.

    A useful param set used by me can be:
    d2_plot_args = {
    'cmap': 'bwr',
    'clim': (-2e-5, 2e-5),
    'method': 'pcolormesh',
    'title': "DSS Data (simulated)",
    'ylabel': "Depth (ft)",
    'clabel': "Strain"
}

d1_plot_args = {
    'ylabel': "Pressure (psi)"
}

    Args:
        data2d (Data2D): The 2D data object to plot on the top subplot.
        data1d (Data1D): The 1D data object to plot on the bottom subplot.
        d2_plot_args (dict, optional): Keyword arguments to override 2D plot defaults.
        d1_plot_args (dict, optional): Keyword arguments to override 1D plot defaults.
        figsize (tuple, optional): The figure size. Defaults to (7, 6).
    """
    # --- Set up Default Arguments ---
    # Defaults for 2D plot
    defaults_2d = {
        'cmap': 'bwr',
        'method': 'pcolormesh',
        'title': data2d.name if data2d.name else "2D Data",
        'ylabel': 'daxis',
        'clabel': 'value' # Custom arg for colorbar label
    }
    if data2d.data is not None and data2d.data.size > 0:
        defaults_2d['clim'] = (np.nanmin(data2d.data), np.nanmax(data2d.data))

    # Defaults for 1D plot
    defaults_1d = {
        'ylabel': 'value' # Custom arg for y-axis label
    }

    # --- Merge User-provided Arguments ---
    # User args take precedence over defaults
    final_d2_args = defaults_2d.copy()
    if d2_plot_args:
        final_d2_args.update(d2_plot_args)

    final_d1_args = defaults_1d.copy()
    if d1_plot_args:
        final_d1_args.update(d1_plot_args)

    # --- Data Preparation ---
    # Crop 1D data to the time range of 2D data
    data1d_cropped = data1d.copy()
    end_time_2d = data2d.get_end_time(time_format='datetime')
    if data2d.start_time is None or end_time_2d is None:
        raise ValueError("data2d must have start_time and taxis set.")
    data1d_cropped.crop(data2d.start_time, end_time_2d)

    # --- Plotting ---
    fig = plt.figure(figsize=figsize)
    ax1 = plt.subplot2grid((5, 4), (0, 0), rowspan=4, colspan=4) # <- 2D plot
    ax2 = plt.subplot2grid((5, 4), (4, 0), rowspan=1, colspan=4, sharex=ax1) # <- 1D plot

    # 2D plot
    d2_clabel = final_d2_args.pop('clabel', None) # Pop custom arg before plotting
    im1 = data2d.plot(ax=ax1, **final_d2_args)
    # title and ylabel are handled by data2d.plot if passed in final_d2_args

    # Hide x-axis ticks for top plot
    ax1.tick_params(labelbottom=False)

    # 1D plot
    d1_ylabel = final_d1_args.pop('ylabel', "Value") # Pop custom arg before plotting
    data1d_cropped.plot(ax=ax2, **final_d1_args)
    ax2.set_ylabel(d1_ylabel) # Set ylabel after plotting

    # Add colorbar for 2D plot
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    cbar = fig.colorbar(im1, cax=cax, orientation='vertical')

    if d2_clabel:
        cbar.set_label(d2_clabel)

    # To align plots, add a dummy axes to the second plot
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="2%", pad=0.05)
    cax2.axis('off')

    plt.tight_layout()
    return fig, (ax1, ax2)

