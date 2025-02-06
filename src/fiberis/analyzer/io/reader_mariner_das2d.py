# Utils for OptaSense - Bakken Mariner - Low Frequency DAS data
# Shenyao Jin, shenyaojin@mines.edu

import h5py
import datetime
import pandas as pd
from glob import glob
import numpy as np
from tqdm import tqdm
from dateutil import parser

from fiberis.analyzer.io import core

class mariner_das2d_io(core.dataio):

    def __init__(self):
        """
        Initialize the mariner_das2d_io object.
        """
        self.daxis = None
        self.taxis = None
        self.data = None
        self.start_time = None

    # Set the data manually

    def set_daxis(self, daxis):
        """
        Set the daxis of the data.

        Parameters:
        ----------
        daxis : numpy.ndarray
            The daxis of the data.
        """
        self.daxis = daxis
        self.record_log(f'daxis is set.')

    def set_taxis(self, taxis):
        """
        Set the taxis of the data.

        Parameters:
        ----------
        taxis : numpy.ndarray
            The taxis of the data.
        """
        self.taxis = taxis
        self.record_log(f'taxis is set.')

    def set_data(self, data):
        """
        Set the data.

        Parameters:
        ----------
        data : numpy.ndarray
            The data.
        """
        self.data = data
        self.record_log(f'data is set.')

    def set_start_time(self, start_time):
        """
        Set the start time of the data.

        Parameters:
        ----------
        start_time : datetime.datetime
            The start time of the data.
        """
        self.start_time = start_time
        self.record_log(f'start_time is set.')

    def self_check(self):
        """
        Check if the data is correctly set.
        """
        if self.daxis is None:
            raise ValueError('daxis is not set.')
        if self.taxis is None:
            raise ValueError('taxis is not set.')
        if self.data is None:
            raise ValueError('data is not set.')
        if self.start_time is None:
            raise ValueError('start_time is not set.')

        # Check the shape of the data shape is correct or not
        # (distance, time)
        if self.data.shape[0] != len(self.daxis):
            raise ValueError('The shape of the data is not correct.')
        if self.data.shape[1] != len(self.taxis):
            raise ValueError('The shape of the data is not correct.')

        # Check the data type
        # datetime for start_time
        if not isinstance(self.start_time, datetime.datetime):
            raise ValueError('start_time is not datetime.datetime.')

    # Read data from folder
    # OptaSense organized data in a folder with each file as a time step

    def read(self, folder_path, start_time=None,end_time=None,
             file_type='h5', verbose=False,
        timestamp_bgind = -34, timestamp_edind = -15,
        filename_timefmt = '%Y%m%d_%H%M%S.%f'):
        """
        Read data from a folder.
        :param folder_path: The path to the folder containing the data.
        :param file_type: The type of the file(s) in the folder.
        :param verbose: Whether to print the log.
        :param timestamp_bgind: The index for the beginning of the timestamp.
        :param timestamp_edind: The index for the end of the timestamp.
        They are determined by the project.
        :param filename_timefmt: The time format in the filename.

        :return: None. The data is stored in the object and ready to pack into npz.
        """

        files = glob(folder_path + '/*.' + file_type)
        files = np.sort(files)

        timestamps = np.array([datetime.datetime.strptime
                               (f[timestamp_bgind:timestamp_edind],
                                filename_timefmt) for f in files])

        # Read all the data. So the start time would be timestamp[0],
        # and end time would be timestamp[-1].

        start_time = timestamps[0]
        end_time = timestamps[-1]

        # set ind (though it is all the data)
        ind = (timestamps >= start_time) & (timestamps <= end_time)

        # Initialize the axis and data
        self.taxis = []
        self.data = []

        flag = 0
        for filename in tqdm(files[ind]):
            # Read h5 file. I also pack this function in io_utils.py
            with h5py.File(filename, 'r') as file:
                # Read the 2D matrix ('RawData') and 1D series ('RawDataTime')
                raw_data = file['Acquisition/Raw[0]/RawData'][:]
                raw_data_time = file['Acquisition/Raw[0]/RawDataTime'][:]

            # Convert 'RawDataTime' to Python datetime objects
            raw_data_time = [datetime.datetime.fromtimestamp(ts / 1e6) for ts in raw_data_time]
            self.record_log(f'Read {filename}.')

            # Append the data to the list
            # Setup the start time and distance axis
            if flag == 0:
                self.start_time = raw_data_time[0]

                flag = 1

        self.record_log(f'Finished reading data from {folder_path}. '
                        f'Need to map the distance axis.')
        return None

    def map_daxis(self, mapping_file):
        """
        Map the distance axis to the data.

        Parameters:
        ----------
        :param mapping_file: The path to the file containing the mapping.
        """

        # Check if the data is correctly set
        if self.data is None:
            raise ValueError('data is not set.')

        chans = np.arange(self.data.shape[0])
        # Read the mapping file
        fiber_map = pd.read_csv(mapping_file, sep = '\t',
                                header=None, names=['channel', 'depth'])
        self.daxis = np.interp(chans, fiber_map['channel'], fiber_map['depth'])