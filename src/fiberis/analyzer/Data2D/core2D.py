from fiberis.utils import signal_utils
import numpy as np
import matplotlib.pyplot as plt
import datetime
from copy import deepcopy, copy
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator


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

    def load_npz(self, filename):

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

        self.taxis = np.array(self.taxis)

        # Find the indices within the range and crop data
        time_mask = (self.taxis >= start) & (self.taxis <= end)
        self.data = self.data[:, time_mask]
        self.taxis = self.taxis[time_mask]
        self.start_time += datetime.timedelta(seconds=start)

        self.taxis -= start
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

        self.daxis = np.array(self.daxis)

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

    def plot(self, ax=None, method='imshow', useTimeStamp=False, xaxis_rotation=0, xtickN=4, *args, **kwargs):
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

        Returns:
        --------
        matplotlib.image.AxesImage or matplotlib.collections.QuadMesh
            The image object for further manipulation (e.g., updating clim).

        Usage:
        ------
        fig, ax = plt.subplots(figsize=(4, 6))
        cx = np.array([-1, 1])
        img1 = DASdata.plot(ax=ax, useTimeStamp=False, cmap='bwr', aspect='auto')
        img1.set_clim(cx * 1000)
        # Set title
        ax.set_title("LF-DAS data")
        plt.show()
        """

        # Remove and handle 'aspect' separately
        aspect = kwargs.pop('aspect', None)

        if useTimeStamp:
            timestamps = [self.start_time + datetime.timedelta(seconds=t) for t in self.taxis]
        else:
            timestamps = self.taxis

        if ax is None:
            fig, ax = plt.subplots()

        # Create the plot based on the selected method
        if method == 'imshow':
            img = ax.imshow(self.data, *args, extent=[timestamps[0], timestamps[-1], self.daxis[-1], self.daxis[0]],
                            **kwargs)
        elif method == 'pcolormesh':
            img = ax.pcolormesh(timestamps, self.daxis, self.data, *args, **kwargs)

        # Set axis labels and handle timestamp formatting
        ax.set_xlabel('Time')
        ax.set_ylabel('Depth')
        if useTimeStamp:
            formatter = mdates.DateFormatter('%m-%d %H:%M')
            ax.xaxis.set_major_formatter(formatter)
            ax.xaxis.set_major_locator(MaxNLocator(xtickN))
            ax.tick_params(axis='x', labelrotation=xaxis_rotation)
            ax.xaxis_date()

        # Apply aspect setting if provided
        if aspect is not None:
            ax.set_aspect(aspect)

        # Add colorbar. Uncomment this. I think it is better to leave it to the user.
        # plt.colorbar(img, ax=ax, label='Amplitude')

        # Display the plot if no axis was provided
        if ax is None:
            plt.show()

        return img

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

    def right_merge(self, data):
        """
        Merge another Data2D instance to the right along the time axis.

        Parameters:
        ----------
        data : Data2D (or subclass)
            The Data2D-compatible instance to merge with the current instance.

        Raises:
        -------
        ValueError
            If the depth axes do not match.
            If the start_time of `data` is earlier than the end time of `self`.

        """
        if not np.array_equal(self.daxis, data.daxis):
            raise ValueError("Depth axes do not match, merging not possible.")

        # Calculate end time of self
        end_time_self = self.start_time + datetime.timedelta(seconds=float(self.taxis[-1]))

        if data.start_time < end_time_self:
            raise ValueError(
                f"Cannot merge: start_time of new data ({data.start_time}) "
                f"is earlier than the end time of existing data ({end_time_self})."
            )

        # Calculate the right shift for the new data
        taxis_shifted = data.taxis + (data.start_time - self.start_time).total_seconds()

        # concatenate the data
        self.taxis = np.concatenate((self.taxis, taxis_shifted))
        self.data = np.concatenate((self.data, data.data), axis=1)

        self.record_log("Merged with", data.name, "at time", data.start_time)

    def get_start_time(self):
        """
        :return the start time of the data
        """
        return self.start_time

    def get_end_time(self, type='datetime'):
        """

        :param type: return type, datetime or seconds
        :return: the end time of the data
        """

        if type == 'datetime':
            return self.start_time + datetime.timedelta(seconds=float(self.taxis[-1]))
        elif type == 'seconds':
            return self.taxis[-1]
        else:
            raise ValueError("type input illegal.")

    def calculate_time(self):
        """
        Calculate the datetime values from the time axis.

        Returns:
        --------
        new_taxis : ndarray of datetime.datetime
            The time axis converted to datetime format.
        """
        if self.start_time is None:
            raise ValueError("start_time is not set.")

        new_taxis = np.array([self.start_time + datetime.timedelta(seconds=t) for t in self.taxis])
        return new_taxis

    def calculate_time_seconds(self):
        """
        Calculate the seconds values from the time axis.
        """
        return self.taxis
    
    def apply_lowpass_filter(self, cutoff_freq, sample_rate=None, order=5):
        """
        Apply a low-pass filter to the data using the bpfilter function.

        Parameters:
        ----------
        cutoff_freq : float
            The cutoff frequency for the low-pass filter
        sample_rate : float, optional
            The sample rate of the data. If None, will be calculated from time axis
        order : int, optional
            The order of the filter (default is 5)

        Returns:
        --------
        None
            Modifies the data in place
        """
        if sample_rate is None:
            # Calculate sample rate from time axis if not provided
            if len(self.taxis) > 1:
                dt = np.mean(np.diff(self.taxis))
                sample_rate = 1 / dt
            else:
                raise ValueError("Cannot calculate sample rate from single time point")
        else:
            dt = 1 / sample_rate

        # Apply low-pass filter to each row of data using bpfilter
        filtered_data = np.zeros_like(self.data)
        for i in range(self.data.shape[0]):
            filtered_data[i] = signal_utils.bpfilter(
                self.data[i],
                dt,
                lowcut=0,  # Set low cutoff to 0 for low-pass behavior
                highcut=cutoff_freq,
                order=order,
                axis=-1
            )

        # Update data and record operation
        self.data = filtered_data
        self.record_log(f"Applied low-pass filter: cutoff={cutoff_freq}Hz, order={order}")

    def get_value_by_depth(self, depth):
        """
        Get the value of a single channel at a specific depth.
        :return: 1D array. Have same length as the time axis.
        """

        # Do check for the depth
        if depth < self.daxis[0] or depth > self.daxis[-1]:
            raise ValueError(f"Depth out of range; {depth} not in {self.daxis[0]} to {self.daxis[-1]}")

        # Find the index of the depth
        depth_idx = np.argmin(np.abs(self.daxis - depth))
        return self.data[depth_idx]