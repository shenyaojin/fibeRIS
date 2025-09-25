# This file contains the core class for data input/output.
# Read data from different types of file(s).
# For large data, it is better to separate the time axis and depth axis to read in. But for save data, should not specify the range.
# The .npz file is recommended for large data, I recommend smaller than 16GB for a single file.
# If the data is larger than 16GB, please split it into multiple files and use `right_merge` function to merge them in analyzer.
# Write data npz(or csv) to make it faster to read.
# Shenyao Jin, shenyaojin@mines.edu

import datetime
from abc import abstractmethod, ABC
from fiberis.utils.history_utils import InfoManagementSystem


class DataIO(ABC):

    # constructor
    def __init__(self):
        """
        Initialize the dataio object. It includes all the common attributes, for the subclasses to inherit.
        """
        self.daxis = None
        self.taxis = None
        self.data = None
        self.start_time = None
        self.filename = None

        # For 3D data and 3D geometry handling
        self.xaxis = None
        self.yaxis = None
        self.zaxis = None

        # Replace the simple list-based history with the InfoManagementSystem.
        # History attributes
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

    @abstractmethod
    def to_analyzer(self):
        """
        Convert the data to an analyzer object for further analysis.

        Returns:
        -------
        analyzer : Analyzer
            The analyzer object.
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