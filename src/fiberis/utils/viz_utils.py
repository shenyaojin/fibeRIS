# Enhanced viz tools for multi-stage hydraulic fracturing
# Shenyao Jin, 02/2025

import os
import matplotlib.pyplot as plt
import numpy as np
from fiberis.io.reader_moose_vpp import MOOSEVectorPostProcessorReader
from fiberis.io.reader_moose_ps import MOOSEPointSamplerReader

# ------ MOOSE related plotting functions ------

# MOOSE post processor plotting function
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

