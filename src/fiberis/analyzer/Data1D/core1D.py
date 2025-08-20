# Define the class Data1D and its methods
# Shenyao Jin, shenyaojin@mines.edu, 02/06/2025
# Description and docs are completed by Gemini, 03/06/2025

import numpy as np
import matplotlib.pyplot as plt
import datetime
from copy import deepcopy
from fiberis.utils import signal_utils, history_utils  # Assuming these are available
from typing import Optional, Union, List, Tuple, Any  # For type hinting
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import os  # For path operations


class Data1D:
    """
    Class for handling one-dimensional time-series data.

    Attributes:
        taxis (Optional[np.ndarray]): Time axis in seconds, relative to start_time.
        data (Optional[np.ndarray]): Data values corresponding to the taxis.
        start_time (Optional[datetime.datetime]): Absolute start time of the data.
        name (Optional[str]): Name or identifier for the data, often the source filename.
        history (history_utils.InfoManagementSystem): System for logging operations.
    """

    def __init__(self, data: Optional[np.ndarray] = None,
                 taxis: Optional[np.ndarray] = None,
                 start_time: Optional[datetime.datetime] = None,
                 name: Optional[str] = None):
        """
        Initialize the Data1D class.

        Args:
            data (Optional[np.ndarray]): Initial data array.
            taxis (Optional[np.ndarray]): Initial time axis (in seconds, relative to start_time).
            start_time (Optional[datetime.datetime]): Initial start time.
            name (Optional[str]): Initial name for the data.
        """
        self.taxis: Optional[np.ndarray] = taxis
        self.data: Optional[np.ndarray] = data
        self.start_time: Optional[datetime.datetime] = start_time
        self.name: Optional[str] = name
        self.history: history_utils.InfoManagementSystem = history_utils.InfoManagementSystem()

        if self.name:
            self.history.add_record(f"Initialized Data1D object with name: {self.name}")
        else:
            self.history.add_record("Initialized empty Data1D object.")

    def load_npz(self, filename: str) -> None:
        """
        Load data from an .npz file and set the class attributes.
        The .npz file is expected to contain 'data', 'taxis', and 'start_time' keys.

        Args:
            filename (str): The path to the .npz file.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            KeyError: If the .npz file is missing one of the required keys.
        """
        self.history.add_record(f"Attempting to load data from NPZ file: {filename}", level="INFO")
        if not os.path.exists(filename):
            self.history.add_record(f"Error: File not found at {filename}", level="ERROR")
            raise FileNotFoundError(f"The specified NPZ file does not exist: {filename}")

        try:
            with np.load(filename, allow_pickle=True) as data_structure:
                if 'data' not in data_structure:
                    raise KeyError("'data' key is missing from the NPZ file.")
                if 'taxis' not in data_structure:
                    raise KeyError("'taxis' key is missing from the NPZ file.")
                if 'start_time' not in data_structure:
                    raise KeyError("'start_time' key is missing from the NPZ file.")

                self.data = data_structure['data']
                self.taxis = data_structure['taxis']
                start_time_raw = data_structure['start_time'].item() # .item() extracts the scalar value

                if isinstance(start_time_raw, str):
                    self.start_time = datetime.datetime.fromisoformat(start_time_raw)
                elif isinstance(start_time_raw, datetime.datetime):
                    self.start_time = start_time_raw
                else:
                    raise ValueError(f"Unsupported start_time type in NPZ file: {type(start_time_raw)}")

            self.name = os.path.basename(filename)
            self.history.add_record(f"Successfully loaded data from {filename}. Name set to '{self.name}'.", level="INFO")
            self.history.add_record(f"Loaded {len(self.data)} data points.", level="INFO")

        except Exception as e:
            self.history.add_record(f"An unexpected error occurred while loading {filename}: {e}", level="ERROR")
            # Re-raise the exception to ensure the calling code knows about the failure.
            raise

    def crop(self, start: Union[datetime.datetime, float, int], end: Union[datetime.datetime, float, int]) -> None:
        """
        Crop the data to a specific time range. Modifies data and taxis in place.
        The time axis (taxis) will be adjusted to start from 0 relative to the new start_time.

        Args:
            start (Union[datetime.datetime, float, int]):
                The start time for cropping. If float/int, interpreted as seconds relative to current start_time.
            end (Union[datetime.datetime, float, int]):
                The end time for cropping. If float/int, interpreted as seconds relative to current start_time.

        Raises:
            TypeError: If start or end types are incompatible or not datetime/float/int.
            ValueError: If start_time is not set, data is not loaded, or start > end.
        """
        if self.start_time is None:
            self.history.add_record("Error: Cannot crop, start_time is not set.", level="ERROR")
            raise ValueError("start_time is not set. Load data first.")
        if self.data is None or self.taxis is None:
            self.history.add_record("Error: Cannot crop, data or taxis is not loaded.", level="ERROR")
            raise ValueError("Data or taxis is not loaded. Load data first.")

        # Convert int to float for uniformity in relative time calculations
        if isinstance(start, int): start = float(start)
        if isinstance(end, int): end = float(end)

        if not (isinstance(start, (datetime.datetime, float)) and isinstance(end, (datetime.datetime, float))):
            self.history.add_record(f"Error: Invalid types for crop start/end ({type(start)}, {type(end)})",
                                    level="ERROR")
            raise TypeError("Start and end must be either datetime.datetime or float (seconds) types.")

        if start > end:
            self.history.add_record(f"Error: Crop start time ({start}) is after end time ({end}).", level="ERROR")
            raise ValueError("Start time must be less than or equal to end time.")

        # Determine crop window in seconds relative to current self.start_time
        start_seconds: float
        end_seconds: float
        if isinstance(start, datetime.datetime) and isinstance(end, datetime.datetime):
            if start < self.start_time:  # Adjust if absolute start is before data's start_time
                start_seconds = 0.0
            else:
                start_seconds = (start - self.start_time).total_seconds()
            end_seconds = (end - self.start_time).total_seconds()
        elif isinstance(start, float) and isinstance(end, float):
            start_seconds = start
            end_seconds = end
        else:
            # This case should be caught by the earlier type check, but as a safeguard:
            self.history.add_record(
                f"Error: Mismatched types for crop start/end after conversion ({type(start)}, {type(end)})",
                level="ERROR")
            raise TypeError("Start and end must be of the same type (datetime or float).")

        if self.taxis.size == 0:  # No data to crop
            # Update start_time to the beginning of the (empty) crop window
            self.start_time += datetime.timedelta(seconds=start_seconds)  # start_seconds is already float
            self.history.add_record(
                f"Cropped an already empty dataset. Start time adjusted. Window: {start_seconds}s to {end_seconds}s.",
                level="WARNING")
            return

        # Find indices within the crop window
        ind = (self.taxis >= start_seconds) & (self.taxis <= end_seconds)

        actual_crop_start_offset_seconds: float  # Ensure this is treated as float
        if not np.any(ind):
            # No data falls within the crop window
            self.data = np.array([])
            self.taxis = np.array([])
            # New start_time is the beginning of the (now empty) crop window
            self.start_time += datetime.timedelta(seconds=start_seconds)  # start_seconds is already float
            self.history.add_record(
                f"Cropped to an empty dataset. Window: {start_seconds}s to {end_seconds}s rel. to original start.",
                level="INFO")
        else:
            # Determine the first time point in the original taxis that is part of the crop
            first_taxis_in_crop = self.taxis[ind][0]
            # Ensure actual_crop_start_offset_seconds is a Python float for timedelta
            actual_crop_start_offset_seconds = float(first_taxis_in_crop)

            self.data = self.data[ind]
            # Adjust taxis to be relative to the new start_time, starting from 0
            # The subtraction result will match the type of self.taxis[ind] or be promoted if necessary.
            self.taxis = self.taxis[ind] - actual_crop_start_offset_seconds
            # Ensure taxis remains or becomes float if actual_crop_start_offset_seconds was float
            if isinstance(actual_crop_start_offset_seconds, float) and not np.issubdtype(self.taxis.dtype, np.floating):
                self.taxis = self.taxis.astype(float)

            # Update start_time to reflect the beginning of the successfully cropped data
            self.start_time += datetime.timedelta(seconds=actual_crop_start_offset_seconds)  # Now uses float
            self.history.add_record(
                f"Data cropped. Original window: [{start_seconds:.2f}s, {end_seconds:.2f}s]. New start time: {self.start_time.isoformat()}",
                level="INFO")

    select_time = crop  # Alias for crop method

    def shift(self, shift_val: Union[datetime.timedelta, float, int]) -> None:
        """
        Apply a time shift to the data by adjusting its start_time.

        Args:
            shift_val (Union[datetime.timedelta, float, int]):
                The time shift to apply. If float/int, interpreted as seconds.
        """
        if self.start_time is None:
            self.history.add_record("Error: Cannot shift, start_time is not set.", level="ERROR")
            raise ValueError("start_time is not set. Load data first.")

        shift_delta: datetime.timedelta
        if isinstance(shift_val, datetime.timedelta):
            shift_delta = shift_val
        elif isinstance(shift_val, (int, float)):
            shift_delta = datetime.timedelta(seconds=float(shift_val))
        else:
            self.history.add_record(f"Error: Invalid type for shift_val ({type(shift_val)})", level="ERROR")
            raise TypeError("Shift value must be either a datetime.timedelta or a float/int representing seconds.")

        self.start_time += shift_delta
        self.history.add_record(
            f"Time shifted by {shift_delta.total_seconds():.3f} seconds. New start_time: {self.start_time.isoformat()}",
            level="INFO")

    def get_value_by_time(self, time_sec: float) -> float:
        """
        Get the interpolated data value at a specific time point.
        Time is relative to the current start_time (i.e., corresponds to a value in self.taxis).

        Args:
            time_sec (float): The time in seconds (relative to self.start_time) at which to get the data value.

        Returns:
            float: The interpolated data value at the specified time.

        Raises:
            ValueError: If data or taxis is not loaded or is empty.

        Note:
            If performance is an issue for many calls, consider cropping the data first.
            This method uses linear interpolation. `np.interp` extrapolates if time_sec is outside taxis range.
        """
        if self.data is None or self.taxis is None or self.data.size == 0 or self.taxis.size == 0:
            self.history.add_record(f"Error: Cannot get value at {time_sec}s, data/taxis is not loaded or empty.",
                                    level="ERROR")
            raise ValueError("Data or taxis is not loaded or is empty.")
        if self.taxis.size == 1:  # Single point data
            if np.isclose(time_sec, self.taxis[0]):
                return float(self.data[0])
            else:  # Extrapolation for single point is just the point itself for np.interp
                return float(np.interp(time_sec, self.taxis, self.data))

        value = np.interp(time_sec, self.taxis, self.data)
        return float(value)

    def calculate_time(self) -> np.ndarray:
        """
        Calculate the absolute datetime values for each point in the time axis.

        Returns:
            np.ndarray: An array of np.datetime64 objects.

        Raises:
            ValueError: If start_time or taxis is not set.
        """
        if self.start_time is None:
            self.history.add_record("Error: Cannot calculate absolute time, start_time is not set.", level="ERROR")
            raise ValueError("start_time is not set.")
        if self.taxis is None:
            self.history.add_record("Error: Cannot calculate absolute time, taxis is not set.", level="ERROR")
            raise ValueError("taxis is not set.")

        try:
            # Ensure self.taxis is float for timedelta64[s] if it's not already, to avoid precision loss with some numpy versions
            taxis_for_timedelta = self.taxis
            if not np.issubdtype(taxis_for_timedelta.dtype, np.floating):
                taxis_for_timedelta = self.taxis.astype(float)

            timedeltas = np.array(taxis_for_timedelta, dtype='timedelta64[s]')
            absolute_np_datetimes = np.datetime64(self.start_time) + timedeltas
            return absolute_np_datetimes
        except Exception as e:
            self.history.add_record(f"Warning: Using fallback for time calculation due to: {e}", level="WARNING")
            # Fallback to ensure seconds are float for timedelta
            new_taxis_dt = np.array([self.start_time + datetime.timedelta(seconds=float(t)) for t in self.taxis])
            return new_taxis_dt

    def copy(self) -> 'Data1D':
        """
        Create a deep copy of the Data1D instance.

        Returns:
            Data1D: A new instance of Data1D with the same data and attributes.
        """
        new_copy = deepcopy(self)
        new_copy.history.add_record(f"Created a deep copy from '{self.name if self.name else 'Unnamed Data1D'}'.")
        return new_copy

    def rename(self, new_name: str) -> None:
        """
        Rename the data instance.

        Args:
            new_name (str): The new name for the data instance.

        Raises:
            TypeError: If the new name is not a string.
            ValueError: If the new name is an empty string or whitespace only.
        """
        if not isinstance(new_name, str):
            self.history.add_record(f"Error: Rename failed, new_name '{new_name}' is not a string.", level="ERROR")
            raise TypeError("The new name must be a string.")

        stripped_name = new_name.strip()
        if not stripped_name:
            self.history.add_record("Error: Rename failed, new_name cannot be empty or whitespace only.", level="ERROR")
            raise ValueError("The new name cannot be empty or whitespace only.")

        old_name = self.name
        self.name = stripped_name
        self.history.add_record(f"Renamed data from '{old_name if old_name else 'Unnamed'}' to '{self.name}'.",
                                level="INFO")

    def get_end_time(self, use_timestamp: bool = False) -> Union[datetime.datetime, np.float64]:

        """
        Get the end time of the data.

        Args:
            use_timestamp (bool): If True, return absolute datetime object.
                                  If False (default), return seconds from start_time.

        Returns:
            Union[datetime.datetime, np.float64]: The end time as a datetime object or seconds.

        Raises:
            ValueError: If start_time or taxis is not set or if taxis is empty.
        """
        if self.start_time is None:
            self.history.add_record("Error: Cannot get end time, start_time is not set.", level="ERROR")
            raise ValueError("start_time is not set.")
        if self.taxis is None:
            self.history.add_record("Error: Cannot get end time, taxis is not set.", level="ERROR")
            raise ValueError("taxis is not set.")
        if self.taxis.size == 0:
            self.history.add_record("Error: Cannot get end time, taxis is empty.", level="ERROR")
            raise ValueError("taxis is empty.")

        last_time_sec = float(self.taxis[-1])
        if use_timestamp:
            end_time = self.start_time + datetime.timedelta(seconds=last_time_sec)
            return end_time
        else:
            return np.float64(last_time_sec)

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None,
             use_timestamp: bool = False, use_legend: bool = True, **kwargs: Any) -> List[Line2D]:
        """
        Plot the data.

        Args:
            ax (Optional[plt.Axes]): Matplotlib axes object to plot on. If None, a new figure and axes are created.
            title (Optional[str]): Title for the plot.
            use_timestamp (bool): If True, x-axis will use absolute datetime objects.
                                  If False (default), x-axis will use seconds from start_time (self.taxis).
            use_legend (bool): If True (default), a legend is displayed.
            **kwargs: Additional keyword arguments to pass to `ax.plot()`.

        Returns:
            List[plt.Line2D]: A list containing the Line2D artists added to the axes.

        Raises:
            ValueError: If data or taxis is not loaded.
        """
        if self.data is None or self.taxis is None:
            self.history.add_record("Error: Cannot plot, data or taxis is not loaded.", level="ERROR")
            raise ValueError("Data or taxis is not loaded. Load data first.")

        if self.taxis.size == 0:
            self.history.add_record("Warning: Plotting empty data.", level="WARNING")
            # Optionally, could draw an empty plot or just return
            if ax is None:
                fig, ax = plt.subplots(figsize=kwargs.pop('figsize', (10, 5)))
                ax.set_title(title if title else "Empty Data")
                ax.set_xlabel("Time")
                ax.set_ylabel("Value")
                if use_legend: ax.legend()
                plt.show()
                return []  # Return empty list of lines
            else:  # ax is provided
                # Potentially do nothing or clear the axes if desired
                return []

        time_axis_to_plot: np.ndarray
        x_label: str
        if use_timestamp:
            time_axis_to_plot = self.calculate_time()  # Returns np.datetime64 array
            x_label = 'Time (Absolute)'
        else:
            time_axis_to_plot = self.taxis
            x_label = f'Time (seconds since {self.start_time.strftime("%Y-%m-%d %H:%M:%S") if self.start_time else "start"})'

        new_figure_created = False
        current_fig = None
        if ax is None:
            current_fig, ax = plt.subplots(figsize=kwargs.pop('figsize', (10, 5)))  # Allow figsize via kwargs
            new_figure_created = True
        else:
            current_fig = ax.get_figure()

        plot_label = self.name if self.name else 'Data'
        # Ensure label is passed if use_legend is True or if a label is provided in kwargs
        if 'label' not in kwargs and use_legend:
            kwargs['label'] = plot_label
        elif 'label' in kwargs and not use_legend:
            pass

        lines = ax.plot(time_axis_to_plot, self.data, **kwargs)

        ax.set_xlabel(x_label)
        ax.set_ylabel('Value')

        if title is not None:
            ax.set_title(title)
        elif new_figure_created and self.name:
            ax.set_title(f"Data: {self.name}")

        if use_legend and (kwargs.get('label') is not None or (plot_label and 'label' not in kwargs)):
            ax.legend(loc=kwargs.pop('legend_loc', 'best'), fontsize=kwargs.pop('legend_fontsize', 'medium'))

        # Ensure time_axis_to_plot is not empty before trying to access its first element
        if use_timestamp and time_axis_to_plot.size > 0 and \
                isinstance(time_axis_to_plot[0], (datetime.datetime, np.datetime64)):
            if current_fig:
                current_fig.autofmt_xdate()

        if new_figure_created:
            plt.show()

        self.history.add_record(f"Plot generated for '{self.name if self.name else 'Unnamed Data1D'}'.", level="INFO")
        return lines

    def print_info(self) -> None:
        """
        Print a summary of the Data1D object's attributes.
        For array attributes (taxis, data), it prints up to the first 10 elements.
        """
        print(f"--- Data1D Object Summary: {self.name or 'Unnamed'} ---")

        print(f"Name: {self.name if self.name else 'Not set'}")
        print(f"Start Time: {self.start_time.isoformat() if self.start_time else 'Not set'}")

        # Data
        if self.data is not None:
            print(f"Data: Length={self.data.shape[0]}")
            if self.data.size > 0:
                if self.data.size < 10:
                    print(f"  Values: {self.data}")
                else:
                    print(f"  Values (first 10): {self.data[:10]}...")
        else:
            print("Data: Not set")

        # Time Axis
        if self.taxis is not None:
            print(f"Time Axis (taxis): Length={self.taxis.shape[0]}")
            if self.taxis.size > 0:
                if self.taxis.size < 10:
                    print(f"  Values: {self.taxis}")
                else:
                    print(f"  Values (first 10): {self.taxis[:10]}...")
        else:
            print("Time Axis (taxis): Not set")

        print(f"History contains {len(self.history.records)} records.")
        print("----------------------------------------------------")

    def right_merge(self, other_data: 'Data1D') -> None:
        """
        Merge another Data1D instance to the right (chronologically) of this instance.
        Modifies self.data and self.taxis in place. Assumes no overlap and other_data starts after self.data ends.

        Args:
            other_data (Data1D): The Data1D instance to merge. Must have compatible data.

        Raises:
            TypeError: If other_data is not a Data1D instance.
            ValueError: If start_times or data/taxis are missing, or if other_data starts before self ends.
        """
        if not isinstance(other_data, Data1D):
            self.history.add_record(f"Error: Merge failed, other_data is not Data1D type ({type(other_data)}).",
                                    level="ERROR")
            raise TypeError("other_data must be an instance of Data1D.")

        if self.start_time is None or other_data.start_time is None:
            self.history.add_record("Error: Merge failed, start_time missing in one or both Data1D objects.",
                                    level="ERROR")
            raise ValueError("Both Data1D objects must have start_time defined.")

        if self.data is None or self.taxis is None or \
                other_data.data is None or other_data.taxis is None:
            self.history.add_record("Error: Merge failed, data/taxis missing in one or both Data1D objects.",
                                    level="ERROR")
            raise ValueError("Both Data1D objects must have data and taxis defined.")

        # Handle cases where one of the datasets is empty
        if self.taxis.size == 0:
            if self.data.size != 0:
                self.history.add_record(
                    "Warning: self.taxis is empty but self.data is not. Correcting self.data to empty.",
                    level="WARNING")
                self.data = np.array([])

            self.data = deepcopy(other_data.data)
            self.taxis = deepcopy(other_data.taxis)
            self.start_time = deepcopy(other_data.start_time)
            self.history.add_record(f"Merged with '{other_data.name}'. Self was empty, copied other_data.",
                                    level="INFO")
            return

        if other_data.taxis.size == 0:
            self.history.add_record(f"Skipped merging with '{other_data.name}' as it is empty.", level="INFO")
            return

            # Calculate end time of current data
        end_time_self = self.start_time + datetime.timedelta(seconds=float(self.taxis[-1]))

        if other_data.start_time < end_time_self:
            msg = (f"Cannot merge: start_time of new data ({other_data.start_time.isoformat()}) "
                   f"is earlier than or overlaps with the end time of existing data ({end_time_self.isoformat()}).")
            self.history.add_record(f"Error: {msg}", level="ERROR")
            raise ValueError(msg)

        # Calculate the time offset for other_data's taxis relative to self.start_time
        time_offset_seconds = (other_data.start_time - self.start_time).total_seconds()
        # Ensure consistent dtype for concatenation, promote to float if needed
        taxis_shifted_other = other_data.taxis + time_offset_seconds
        if not np.issubdtype(self.taxis.dtype, np.floating) and isinstance(time_offset_seconds, float):
            self.taxis = self.taxis.astype(float)
        if not np.issubdtype(taxis_shifted_other.dtype, np.floating) and isinstance(time_offset_seconds, float):
            taxis_shifted_other = taxis_shifted_other.astype(float)

        # If one is float and other is not, make both float before concat
        if np.issubdtype(self.taxis.dtype, np.floating) and not np.issubdtype(taxis_shifted_other.dtype, np.floating):
            taxis_shifted_other = taxis_shifted_other.astype(float)
        elif not np.issubdtype(self.taxis.dtype, np.floating) and np.issubdtype(taxis_shifted_other.dtype, np.floating):
            self.taxis = self.taxis.astype(float)

        self.data = np.concatenate((self.data, other_data.data))
        self.taxis = np.concatenate((self.taxis, taxis_shifted_other))

        self.history.add_record(f"Successfully merged with '{other_data.name}'. New data length: {self.data.size}",
                                level="INFO")

    def remove_abnormal_data(self, threshold: float = 300.0, method: str = 'mean') -> None:
        """
        Remove/replace abnormal data points, often spikes caused by device errors.
        An abnormal point is identified if the absolute difference to its neighbors on both sides exceeds the threshold.
        Modifies self.data in place.

        Args:
            threshold (float): The difference threshold to identify an abnormal data point.
            method (str): Method to replace abnormal data. Options:
                          'mean': Replace with the mean of the two adjacent points.
                          'interp': Replace using linear interpolation from valid neighbors.
                          'nan': Replace with np.nan (useful for later processing).

        Raises:
            ValueError: If data is not loaded or is empty, or if method is invalid.
        """
        if self.data is None or self.data.size == 0:
            self.history.add_record("Error: Cannot remove abnormal data, data is not loaded or empty.", level="ERROR")
            raise ValueError("Data is not loaded or is empty.")

        if self.data.size < 3:
            self.history.add_record("Skipped abnormal data removal: not enough data points (need at least 3).",
                                    level="INFO")
            return

            # Assuming signal_utils.samediff exists and works as intended.
        # If not, a manual diff would be:
        # diff_fwd = np.diff(self.data)
        # diff_bwd = -np.diff(self.data[::-1])[::-1] # or similar logic

        abnormal_idx_list = []
        for i in range(1, len(self.data) - 1):
            diff_to_prev = np.abs(self.data[i] - self.data[i - 1])
            diff_to_next = np.abs(self.data[i + 1] - self.data[i])
            if diff_to_prev > threshold and diff_to_next > threshold:
                abnormal_idx_list.append(i)

        abnormal_idx = np.array(abnormal_idx_list, dtype=int)

        if abnormal_idx.size == 0:
            self.history.add_record("No abnormal data points found with current threshold.", level="INFO")
            return

        clean_data = self.data.copy()  # Operate on a copy
        num_removed = abnormal_idx.size

        if method == 'mean':
            for idx in abnormal_idx:
                clean_data[idx] = (self.data[idx - 1] + self.data[idx + 1]) / 2.0
        elif method == 'interp':
            # Create an array of all indices
            all_indices = np.arange(len(self.data))
            # Get indices of valid (non-abnormal) points
            valid_idx = np.setdiff1d(all_indices, abnormal_idx, assume_unique=True)

            if valid_idx.size < 2:
                self.history.add_record(
                    f"Warning: Not enough valid points ({valid_idx.size}) to interpolate abnormal data. Using 'nan' instead.",
                    level="WARNING")
                method = 'nan'
            else:
                # Interpolate only at abnormal_idx using values from valid_idx
                clean_data[abnormal_idx] = np.interp(abnormal_idx, valid_idx, self.data[valid_idx])

        if method == 'nan':
            clean_data[abnormal_idx] = np.nan

        elif method not in ['mean', 'interp', 'nan']:
            self.history.add_record(f"Error: Invalid method '{method}' for remove_abnormal_data.", level="ERROR")
            raise ValueError(f"Invalid method: {method}. Choose 'mean', 'interp', or 'nan'.")

        self.data = clean_data  # Assign the modified copy back to self.data
        self.history.add_record(
            f"Removed/replaced {num_removed} abnormal data point(s) using '{method}' method with threshold {threshold}.",
            level="INFO")

    def interpolate(self, new_taxis: Union[np.ndarray, List[float]],
                    new_start_time: Optional[datetime.datetime] = None,
                    fill_value_left: Optional[Any] = np.nan,  # Allow 'extrapolate'
                    fill_value_right: Optional[Any] = np.nan  # Allow 'extrapolate'
                    ) -> None:
        """
        Interpolate the data to a new time axis. Modifies data, taxis, and start_time in place.

        Args:
            new_taxis (Union[np.ndarray, List[float]]):
                The new time axis (in seconds) for interpolation. These times are relative
                to `new_start_time` if provided, otherwise relative to the current `self.start_time`.
                Must be monotonically increasing.
            new_start_time (Optional[datetime.datetime]):
                The new absolute start time for the interpolated data. If None, the current
                `self.start_time` is used as the reference for `new_taxis`.
            fill_value_left (Optional[Any]): Value to use for points before the first original data point.
                                             Defaults to np.nan. Can be a float or 'extrapolate'.
            fill_value_right (Optional[Any]): Value to use for points after the last original data point.
                                              Defaults to np.nan. Can be a float or 'extrapolate'.


        Raises:
            ValueError: If data/taxis is not loaded or empty, if new_taxis is empty or not monotonic.
        """
        if self.data is None or self.taxis is None or self.data.size == 0 or self.taxis.size == 0:
            self.history.add_record("Error: Cannot interpolate, current data/taxis is not loaded or empty.",
                                    level="ERROR")
            raise ValueError("Current data or taxis is not loaded or is empty.")

        if self.start_time is None:
            self.history.add_record("Error: Cannot interpolate, current start_time is not set.", level="ERROR")
            raise ValueError("Current start_time is not set.")

        if not isinstance(new_taxis, np.ndarray):
            new_taxis_np = np.array(new_taxis, dtype=float)
        else:
            new_taxis_np = new_taxis.astype(float, copy=True)

        if new_taxis_np.size == 0:
            self.history.add_record("Error: Cannot interpolate to an empty new_taxis.", level="ERROR")
            raise ValueError("new_taxis cannot be empty.")

        if not np.all(np.diff(new_taxis_np) >= 0):
            self.history.add_record("Error: new_taxis must be monotonically increasing for interpolation.",
                                    level="ERROR")
            raise ValueError("new_taxis must be monotonically increasing for interpolation.")

        final_new_start_time: datetime.datetime = new_start_time if new_start_time is not None else self.start_time
        time_offset_for_interp_points = (final_new_start_time - self.start_time).total_seconds()
        interpolation_points_in_original_frame = new_taxis_np + time_offset_for_interp_points

        if not np.all(np.diff(self.taxis) >= 0):
            self.history.add_record("Error: Cannot interpolate, current self.taxis is not monotonically increasing.",
                                    level="ERROR")
            raise ValueError("Original taxis must be monotonically increasing for interpolation.")

        # Handle 'extrapolate' for fill_value
        interp_kwargs = {}
        if fill_value_left == 'extrapolate' or fill_value_right == 'extrapolate':
            from scipy.interpolate import interp1d  # Use scipy for extrapolation
            # Scipy's interp1d requires at least 2 points in the original data
            if self.taxis.size < 2:
                self.history.add_record("Error: Extrapolation requires at least 2 data points in original taxis.",
                                        level="ERROR")
                raise ValueError("Extrapolation requires at least 2 data points in original taxis.")

            interp_func = interp1d(self.taxis, self.data, kind='linear',
                                   fill_value="extrapolate", bounds_error=False)
            interpolated_data = interp_func(interpolation_points_in_original_frame)
        else:
            # Use numpy.interp for non-extrapolating cases or fixed fill values
            interpolated_data = np.interp(
                interpolation_points_in_original_frame,
                self.taxis,
                self.data,
                left=fill_value_left if fill_value_left != 'extrapolate' else None,
                # np.interp uses None for boundary values
                right=fill_value_right if fill_value_right != 'extrapolate' else None
            )
            # If fill_value was a specific float, np.interp handles it via left/right.
            # If it was np.nan (default), it's also handled.

        self.data = interpolated_data
        self.taxis = new_taxis_np
        self.start_time = final_new_start_time

        self.history.add_record(
            f"Data interpolated to new time axis. New start: {self.start_time.isoformat()}, {self.data.size} points.",
            level="INFO"
        )

    def savez(self, filename: str) -> None:
        """
        Save the current data, taxis, and start_time to an .npz file.

        Args:
            filename (str): The path to the .npz file where data will be saved.

        Raises:
            ValueError: If data or taxis is not loaded or empty.
        """
        if self.data is None or self.taxis is None or self.data.size == 0 or self.taxis.size == 0:
            self.history.add_record("Error: Cannot save, data or taxis is not loaded or is empty.", level="ERROR")
            raise ValueError("Data or taxis is not loaded or is empty.")

        np.savez(filename, data=self.data, taxis=self.taxis, start_time=self.start_time.isoformat())
        self.history.add_record(f"Data saved to {filename}.", level="INFO")
