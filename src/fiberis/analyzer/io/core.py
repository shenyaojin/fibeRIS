# This file contains the core class for data input/output.
# Read data from different types of file(s).
# Write data npz(or csv) to make it faster to read.

import importlib
import pkgutil
import os
import datetime
from abc import abstractmethod


class DataIO:

    # constructor
    def __init__(self):
        """
        Initialize the dataio object.
        """
        self.daxis = None
        self.taxis = None
        self.data = None
        self.start_time = None
        self._history = []

    @abstractmethod
    def read(self, **kwargs):
        """
        Read data from a file.

        Parameters:
        ----------
        filename : str
            The path to the file containing the data.
        """
        pass

    @abstractmethod
    def write(self, filename, *args):
        """
        Write data to a file.

        Parameters:
        ----------
        filename : str
            The path to the file to write the data.
        data : numpy.ndarray
            The data to write.
        """
        pass

    # History recording
    def record_log(self, *args):
        time_now = datetime.datetime.now()
        # Concatenate all arguments
        msg = " ".join(map(str, args)) + f" | Time: {time_now}"
        # Append the formatted message to the history
        self.history.append(msg)

    def print_log(self):
        for msg in self.history:
            print(msg)

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