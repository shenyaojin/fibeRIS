from fiberis.analyzer.utils import signal_utils
import numpy as np
import matplotlib.pyplot as plt
import datetime
from copy import deepcopy, copy
import matplotlib.dates as mdates


class Data2D():

    def __init__(self):
        self.data = None
        self.taxis = None
        self.daxis = None
        self.start_time = None
        self.name = None
        self._history = []

    # I/O
    # Manually set the data
    def set_data(self, data):
        """
        Set the data attribute of the Data2D instance.

        Parameters:
        ----------
        data : numpy.ndarray
            The data array to be set for the instance.

        Usage:
        ------
        >>> instance = Data2D()
        >>> instance.set_data(np.array([[1, 2, 3], [4, 5, 6]]))
        """
        self.data = data

    def set_taxis(self, taxis):
        """
        Set the time axis attribute of the Data2D instance.

        Parameters:
        ----------
        taxis : numpy.ndarray
            The array representing time values.

        Usage:
        ------
        >>> instance = Data2D()
        >>> instance.set_taxis(np.array([0, 1, 2, 3]))
        """
        self.taxis = taxis

    def set_daxis(self, daxis):
        """
        Set the depth axis attribute of the Data2D instance.

        Parameters:
        ----------
        daxis : numpy.ndarray
            The array representing depth values.

        Usage:
        ------
        >>> instance = Data2D()
        >>> instance.set_daxis(np.array([10, 20, 30]))
        """
        self.daxis = daxis

    def set_start_time(self, start_time):
        """
        Set the start time attribute of the Data2D instance and validate its type.

        Parameters:
        ----------
        start_time : datetime.datetime
            The start time to be set.

        Raises:
        -------
        TypeError
            If the input is not a datetime object.

        Usage:
        ------
        >>> import datetime
        >>> instance = Data2D()
        >>> instance.set_start_time(datetime.datetime.now())
        """
        if not isinstance(start_time, datetime.datetime):
            raise TypeError("start_time must be a datetime object")
        self.start_time = start_time

    def set_filename(self, *args):
        """
        Set the filename attribute by joining the provided arguments.

        Parameters:
        ----------
        args : str
            Parts of the filename to be joined.

        Usage:
        ------
        >>> instance = Data2D()
        >>> instance.set_filename("file", "2025", "data")
        >>> print(instance.name)  # Output: file_2025_data
        """
        self.name = "_".join(args)

    def read_npz(self, filename):

        if filename[-4:] != ".npz":
            filename += ".npz"

        data_structure = np.load(filename, allow_pickle=True)
        self.data = data_structure['data']
        self.taxis = data_structure['taxis']
        self.daxis = data_structure['daxis']
        self.name = filename
        # Handle start_time correctly if it's an array or numpy type

        start_time_raw = data_structure['start_time']
        if isinstance(start_time_raw, np.ndarray) and start_time_raw.size == 1:
            start_time_raw = start_time_raw.item()

        if isinstance(start_time_raw, np.datetime64):
            self.start_time = start_time_raw.astype('M8[ms]').astype(datetime.datetime)
        elif isinstance(start_time_raw, str):
            self.start_time = datetime.datetime.fromisoformat(start_time_raw)
        else:
            self.start_time = start_time_raw

        self.record_log("Load data from", filename)

    # Data selection by time range with cropping functionality
    def select_time(self, start, end):
        """
        Crop the data within a specific time range.

        Parameters:
        ----------
        start : datetime, float, or int
            The start time for cropping.
        end : datetime, float, or int
            The end time for cropping.

        Returns:
        --------
        None
        """
        if isinstance(start, datetime.datetime):
            start = (start - self.start_time).total_seconds()
        if isinstance(end, datetime.datetime):
            end = (end - self.start_time).total_seconds()

        if not (isinstance(start, (int, float)) and isinstance(end, (int, float))):
            raise TypeError("Start and end must be either datetime, float, or int types.")

        if start > end:
            raise ValueError("Start time must be less than or equal to end time.")

        # Find the indices within the range and crop data
        time_mask = (self.taxis >= start) & (self.taxis <= end)
        self.data = self.data[:, time_mask]
        self.taxis = self.taxis[time_mask]
        self.start_time += datetime.timedelta(seconds=start)

        self.record_log("Time range selected:", start, end)

    # Data selection by depth range with cropping functionality
    def select_depth(self, start, end):
        """
        Crop the data within a specific depth range.

        Parameters:
        ----------
        start : float or int
            The start depth for cropping.
        end : float or int
            The end depth for cropping.

        Returns:
        --------
        None
        """
        if not (isinstance(start, (int, float)) and isinstance(end, (int, float))):
            raise TypeError("Start and end must be either float or int types.")

        if start > end:
            raise ValueError("Start depth must be less than or equal to end depth.")

        # Find the indices within the range and crop data
        depth_mask = (self.daxis >= start) & (self.daxis <= end)
        self.data = self.data[depth_mask, :]
        self.daxis = self.daxis[depth_mask]

        self.record_log("Depth range selected:", start, end)

    def __copy__(self):
        """
        Create a shallow copy of the Data2D instance.

        Returns:
        --------
        Data2D
            A shallow copy of the instance.

        Usage:
        ------
        >>> import copy
        >>> instance = Data2D()
        >>> new_instance = copy.copy(instance)
        """
        return copy(self)

    def copy(self):
        """
        Create a deep copy of the Data2D instance.

        Returns:
        --------
        Data2D
            A deep copy of the instance.

        Usage:
        ------
        >>> instance = Data2D()
        >>> new_instance = instance.copy()
        """
        return deepcopy(self)

    def rename(self, new_name):
        """
        Rename the data instance.

        Parameters:
        ----------
        new_name : str
            The new name for the data instance.

        Raises:
        -------
        TypeError
            If the new name is not a string.
        ValueError
            If the new name is an empty string.
        """
        if not isinstance(new_name, str):
            raise TypeError("The new name must be a string.")
        if not new_name.strip():
            raise ValueError("The new name cannot be empty or whitespace only.")

        self.name = new_name.strip()

    # msg system
    # History recording
    def record_log(self, *args):
        """
        Record a log entry with the current time and provided message arguments.

        Parameters:
        ----------
        args : str
            The message components to be logged.

        Usage:
        ------
        >>> instance = Data2D()
        >>> instance.record_log("Start process", "with parameter X")
        """
        time_now = datetime.datetime.now()
        msg = " ".join(map(str, args)) + f" | Time: {time_now}"
        self._history.append(msg)

    def print_log(self):
        """
        Print all recorded log entries.

        Usage:
        ------
        >>> instance = Data2D()
        >>> instance.print_log()
        """
        for msg in self._history:
            print(msg)

    def plot(self, ax=None, method='imshow', useTimeStamp=False, *args, **kwargs):
        """
        Plot the data using the specified method. Optionally, use real timestamps for the x-axis.

        Parameters:
        ----------
        ax : matplotlib.axes.Axes, optional
            An existing matplotlib axis to plot on. If not provided, a new figure will be created.
        method : str, optional
            The method to use for plotting. Options are 'imshow' or 'pcolormesh'. Default is 'imshow'.
        useTimeStamp : bool, optional
            If True, convert the x-axis to real timestamps based on the start_time attribute.
        *args : tuple
            Additional positional arguments for the matplotlib plotting function.
        **kwargs : dict
            Additional keyword arguments for the matplotlib plotting function (e.g., cmap, extent).

        Usage:
        ------
        >>> instance = DSS2D()
        >>> fig, ax = plt.subplots()
        >>> instance.plot(ax=ax, method='pcolormesh', useTimeStamp=True, cmap='viridis')
        >>> plt.show()
        """

        if useTimeStamp:
            timestamps = [self.start_time + datetime.timedelta(seconds=t) for t in self.taxis]
        else:
            timestamps = self.taxis

        if ax is None:
            plt.figure()
            if method == 'imshow':
                plt.imshow(self.data, *args, extent=[timestamps[0], timestamps[-1], self.daxis[-1], self.daxis[0]], **kwargs)
            elif method == 'pcolormesh':
                plt.pcolormesh(timestamps, self.daxis, self.data, *args, **kwargs)
            plt.colorbar(label='Amplitude')
            plt.xlabel('Time')
            plt.ylabel('Depth')
            if useTimeStamp:
                plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
                plt.gcf().autofmt_xdate()
            plt.show()
        else:
            if method == 'imshow':
                ax.imshow(self.data, *args, extent=[timestamps[0], timestamps[-1], self.daxis[-1], self.daxis[0]], **kwargs)
            elif method == 'pcolormesh':
                ax.pcolormesh(timestamps, self.daxis, self.data, *args, **kwargs)
            if useTimeStamp:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
                ax.figure.autofmt_xdate()

    def shift(self, shift):
        """
        Apply a time shift to the data.

        Parameters:
        ----------
        shift : timedelta or float
            The time shift to apply.
        """
        if isinstance(shift, datetime.timedelta):
            shift_seconds = shift
        elif isinstance(shift, (int, float)):
            shift_seconds = datetime.timedelta(seconds=shift)
        else:
            raise TypeError("Shift must be either a timedelta or a float representing seconds.")

        self.start_time += shift_seconds
