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