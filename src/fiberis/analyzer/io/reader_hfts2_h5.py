# Utils for reading HFTS2 data from h5 files
# Shenyao Jin, adopted from Peiyao Li, Ge Jin's codes

import h5py
import datetime
import numpy as np
from glob import glob
from tqdm import tqdm
from dateutil import parser

from fiberis.analyzer.io import core

# Need to self check function in core.py

class HFTS2DAS2D(core.DataIO):


    def read(self, folderpath):

        # Get folder info
        files = glob(folderpath + '/*.h5')
        files = np.sort(files)
        timestamps = np.array([parser.parse(f[-21:-4]) for f in files])

        # Check if start_time or end_time is a string and convert to datetime
        for file in files:
            raw_data, raw_datga_time = self.read_h5(file)
            # Write it to tmp npz file, then combine them
            

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
            raw_data = file['Acquisition/Raw[0]/RawData'][:]
            raw_data_time = file['Acquisition/Raw[0]/RawDataTime'][:]

        # Convert 'RawDataTime' to Python datetime objects
        raw_data_time = [datetime.datetime.fromtimestamp(ts / 1e6) for ts in raw_data_time]

        return raw_data, raw_data_time