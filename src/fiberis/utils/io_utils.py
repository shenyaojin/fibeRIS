# Toolkit reading files. Might be useful for other projects so I put it here.
import h5py
import datetime

# Modified from Jin's OptaSense DAS processing code
def read_h5(filename):
    """
    Reads a 2D matrix and a 1D datetime series from an HDF5 file and converts the time to Python datetime objects.

    Parameters:
    filename (str): Path to the HDF5 file.

    Returns:
    tuple: A tuple containing the following elements:
        - 2D numpy array: The sensor data.
        - 1D list: The timestamps corresponding to the sensor data, converted to Python datetime objects.
    """
    with h5py.File(filename, 'r') as file:
        # Read the 2D matrix ('RawData') and 1D series ('RawDataTime')
        # from the HDF5 file, this works for the OptaSense DAS files.
        raw_data = file['Acquisition/Raw[0]/RawData'][:]
        raw_data_time = file['Acquisition/Raw[0]/RawDataTime'][:]

    # Convert 'RawDataTime' to Python datetime objects
    raw_data_time = [datetime.datetime.fromtimestamp(ts / 1e6) for ts in raw_data_time]

    return raw_data, raw_data_time