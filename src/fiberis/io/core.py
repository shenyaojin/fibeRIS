# This file contains the core class for data input/output.
# Read data from different types of file(s).
# Write data npz(or csv) to make it faster to read.
# Refactored to use InfoManagementSystem for logging.

import datetime
from abc import abstractmethod, ABC
from fiberis.utils.history_utils import InfoManagementSystem


class DataIO(ABC):

    # constructor
    def __init__(self):
        """
        Initialize the dataio object.
        """
        self.daxis = None
        self.taxis = None
        self.data = None
        self.start_time = None
        self.filename = None
        # Replace the simple list-based history with the InfoManagementSystem
        self.log_system = InfoManagementSystem()

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
    def record_log(self, *args, level: str = "INFO"):
        """
        Records a log message using the InfoManagementSystem.

        This method replaces the previous implementation that manually handled
        timestamps and history lists.

        Parameters:
        ----------
        *args :
            One or more objects to be converted to strings and concatenated
            to form the log message.
        level : str, optional
            The severity level of the log (e.g., "INFO", "WARNING", "ERROR").
            Defaults to "INFO".
        """
        # Concatenate all arguments into a single message string
        msg = " ".join(map(str, args))
        # Add the record to the logging system
        self.log_system.add_record(msg, level=level)

    def print_log(self):
        """
        Prints all recorded logs to the console using the InfoManagementSystem.
        """
        self.log_system.print_records()

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