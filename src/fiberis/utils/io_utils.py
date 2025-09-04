# Toolkit reading files. Might be useful for other projects so I put it here.
import h5py

# Dictionary of possible HDF5 keys for DAS data components.
# The function will iterate through these keys to find the correct ones.
H5_KEYS = {
    'data': ['data', 'Acquisition/Raw[0]/RawData'],
    'daxis': ['depth'],
    'taxis': ['stamps_unix', 'Acquisition/Raw[0]/RawDataTime'],
    'start_time': ['stamps']
}


# Modified from Jin's OptaSense DAS processing code.
def read_h5(filename):
    """
    Reads data, axes, and metadata from an HDF5 file by searching for a set of possible keys.

    Parameters:
    filename (str): Path to the HDF5 file.

    Returns:
    tuple: A tuple containing the following elements:
        - data (numpy.ndarray): The 2D sensor data. (None if not found)
        - daxis (numpy.ndarray): The 1D depth/distance axis. (None if not found)
        - taxis (numpy.ndarray): The 1D raw timestamps. (None if not found)
        - start_time: The acquisition start time, type depends on H5 file. (None if not found)
    """
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
