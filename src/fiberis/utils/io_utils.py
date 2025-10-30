# Toolkit reading files. Might be useful for other projects so I put it here. -- Shenyao

import h5py
import numpy as np
from typing import Optional
# Dictionary of possible HDF5 keys for DAS data components.
# The function will iterate through these keys to find the correct ones.



# Modified from Jin's OptaSense DAS processing code.
def read_h5(filename: str):
    """
    Reads data, axes, and metadata from an HDF5 file by searching for a set of possible keys. Only use it to load
    DAS data. For other type of h5 file, I'm finding ways to load it. -- Shenyao

    Parameters:
    filename (str): Path to the HDF5 file.

    Returns:
    tuple: A tuple containing the following elements:
        - data (numpy.ndarray): The 2D sensor data. (None if not found)
        - daxis (numpy.ndarray): The 1D depth/distance axis. (None if not found)
        - taxis (numpy.ndarray): The 1D raw timestamps. (None if not found)
        - start_time: The acquisition start time, type depends on H5 file. (None if not found)
    """
    H5_KEYS = {
        'data': ['data', 'Acquisition/Raw[0]/RawData'],
        'daxis': ['depth'],
        'taxis': ['stamps_unix', 'Acquisition/Raw[0]/RawDataTime'],
        'start_time': ['stamps']
    }

    data = None
    daxis = None
    taxis = None
    start_time = None

    with h5py.File(filename, 'r') as file:
        # --- Read Data (2D matrix) ---
        for key in H5_KEYS['data']:
            try:
                data = file[key][:]
                break
            except KeyError:
                continue

        # --- Read Depth Axis (daxis) ---
        for key in H5_KEYS['daxis']:
            try:
                daxis = file[key][:]
                break
            except KeyError:
                continue

        # --- Read Time Axis (taxis) ---
        for key in H5_KEYS['taxis']:
            try:
                taxis = file[key][:]
                break
            except KeyError:
                continue

        # --- Read Start Time ---
        for key in H5_KEYS['start_time']:
            try:
                start_time = file[key][0]
                break
            except KeyError:
                continue

    return data, daxis, taxis, start_time


def load_hfts2_depthtable(filename: str) -> np.ndarray:
    """
    Reads a HFTS2 DAS depth table from a .csvh file.

    Parameters:
    filename (str): Path to the .csvh file.

    Returns:
    numpy.ndarray: A 1D array of depth values.
    """
    depths = []
    in_data_section = False
    data_header_lines_to_skip = 2

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.startswith('~DATA'):
                in_data_section = True
                continue

            if in_data_section:
                if data_header_lines_to_skip > 0:
                    data_header_lines_to_skip -= 1
                    continue

                parts = line.split(',')
                if len(parts) >= 2:
                    try:
                        # The value is in the second column, e.g., "1,-317.911398"
                        depth_val_str = parts[1].strip()
                        depths.append(float(depth_val_str))
                    except (ValueError, IndexError):
                        print(f"Could not parse depth value from line: '{line}'. Skipping.")
                        continue

    return np.array(depths)

def load_h5_by_group_name(filepath: str, group_name: str) -> np.ndarray:
    """
    Load the h5 file using group name as a key value. Used for those pumping curve files (Why not use csv...?) OR other
    h5 files I did not design a I/O
    --Shenyao

    :param filepath: the filepath of the h5 file
    :param group_name: the key value, looks like 'Acquisition/Raw[0]/RawData'.
    :return: the np.array of the loaded data
    """
    with h5py.File(filepath, 'r') as file:
        try:
            data = file[group_name][:]
            return data
        except KeyError:
            print(f"Group '{group_name}' not found in file '{filepath}'. Returning an empty array.")
            return np.array([])


def list_h5_keys(filepath: str) -> list:
    """
    Lists all the keys (group and dataset names) in an HDF5 file.

    :param filepath: The path to the H5 file.
    :return: A list of strings, where each string is a key.
    """
    keys = []
    with h5py.File(filepath, 'r') as file:
        file.visit(keys.append)
    return keys