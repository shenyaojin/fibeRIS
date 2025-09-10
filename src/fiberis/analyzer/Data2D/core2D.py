# Define the class Data2D and its methods
# Original Author: Shenyao Jin, shenyaojin@mines.edu
# Improved by Gemini, 03/06/2025

import numpy as np
import matplotlib.pyplot as plt
import datetime
from copy import deepcopy
# Assuming history_utils.InfoManagementSystem is in the same package path
from fiberis.utils import signal_utils  # Keep this
from fiberis.utils.history_utils import InfoManagementSystem  # More specific import

import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from typing import Optional, Union, List, Tuple, Any, cast, Callable, Dict
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.collections import QuadMesh
import os


class Data2D:
    """
    Class for handling two-dimensional data, typically with time and depth axes.

    Attributes:
        data (Optional[np.ndarray]): 2D array of data values. Expected shape (n_depth_points, n_time_points).
        taxis (Optional[np.ndarray]): 1D array for the time axis (in seconds, relative to start_time).
        daxis (Optional[np.ndarray]): 1D array for the depth/spatial axis.
        start_time (Optional[datetime.datetime]): Absolute start time of the data.
        name (Optional[str]): Name or identifier for the data.
        history (InfoManagementSystem): System for logging operations.
    """

    def __init__(self,
                 data: Optional[np.ndarray] = None,
                 taxis: Optional[np.ndarray] = None,
                 daxis: Optional[np.ndarray] = None,
                 start_time: Optional[datetime.datetime] = None,
                 name: Optional[str] = None):
        """
        Initialize the Data2D class.

        Args:
            data (Optional[np.ndarray]): Initial 2D data array.
            taxis (Optional[np.ndarray]): Initial time axis (seconds relative to start_time).
            daxis (Optional[np.ndarray]): Initial depth/spatial axis.
            start_time (Optional[datetime.datetime]): Initial absolute start time.
            name (Optional[str]): Initial name for the data.
        """
        self.data: Optional[np.ndarray] = data
        self.taxis: Optional[np.ndarray] = taxis
        self.daxis: Optional[np.ndarray] = daxis
        self.start_time: Optional[datetime.datetime] = start_time
        self.name: Optional[str] = name
        self.history: InfoManagementSystem = InfoManagementSystem()

        if self.name:
            self.history.add_record(f"Initialized Data2D object with name: {self.name}", level="INFO")
        else:
            self.history.add_record("Initialized empty Data2D object.", level="INFO")

        # Basic validation if data is provided
        if self.data is not None:
            if self.taxis is not None and self.data.shape[1] != self.taxis.shape[0]:
                msg = f"Data shape[1] ({self.data.shape[1]}) does not match taxis length ({self.taxis.shape[0]})."
                self.history.add_record(f"Initialization Warning: {msg}", level="WARNING")
            if self.daxis is not None and self.data.shape[0] != self.daxis.shape[0]:
                msg = f"Data shape[0] ({self.data.shape[0]}) does not match daxis length ({self.daxis.shape[0]})."
                self.history.add_record(f"Initialization Warning: {msg}", level="WARNING")

    # --- I/O Methods ---
    def set_data(self, data: np.ndarray) -> None:
        """
        Set the data attribute. Validates if taxis and daxis dimensions match if they exist.

        Args:
            data (np.ndarray): The 2D data array.
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            self.history.add_record("Error: Data must be a 2D NumPy array.", level="ERROR")
            raise TypeError("Data must be a 2D NumPy array.")

        if self.taxis is not None and data.shape[1] != self.taxis.shape[0]:
            msg = f"New data's time dimension ({data.shape[1]}) does not match existing taxis length ({self.taxis.shape[0]})."
            self.history.add_record(f"Error: {msg}", level="ERROR")
            raise ValueError(msg)
        if self.daxis is not None and data.shape[0] != self.daxis.shape[0]:
            msg = f"New data's depth dimension ({data.shape[0]}) does not match existing daxis length ({self.daxis.shape[0]})."
            self.history.add_record(f"Error: {msg}", level="ERROR")
            raise ValueError(msg)

        self.data = data
        self.history.add_record("Data attribute updated.", level="INFO")

    def set_taxis(self, taxis: np.ndarray) -> None:
        """
        Set the time axis. Validates if data dimension matches if data exists.

        Args:
            taxis (np.ndarray): 1D array for the time axis (seconds).
        """
        if not isinstance(taxis, np.ndarray) or taxis.ndim != 1:
            self.history.add_record("Error: taxis must be a 1D NumPy array.", level="ERROR")
            raise TypeError("taxis must be a 1D NumPy array.")
        if self.data is not None and self.data.shape[1] != taxis.shape[0]:
            msg = f"New taxis length ({taxis.shape[0]}) does not match existing data's time dimension ({self.data.shape[1]})."
            self.history.add_record(f"Error: {msg}", level="ERROR")
            raise ValueError(msg)

        self.taxis = taxis
        self.history.add_record("Time axis (taxis) updated.", level="INFO")

    def set_daxis(self, daxis: np.ndarray) -> None:
        """
        Set the depth/spatial axis. Validates if data dimension matches if data exists.

        Args:
            daxis (np.ndarray): 1D array for the depth/spatial axis.
        """
        if not isinstance(daxis, np.ndarray) or daxis.ndim != 1:
            self.history.add_record("Error: daxis must be a 1D NumPy array.", level="ERROR")
            raise TypeError("daxis must be a 1D NumPy array.")
        if self.data is not None and self.data.shape[0] != daxis.shape[0]:
            msg = f"New daxis length ({daxis.shape[0]}) does not match existing data's depth dimension ({self.data.shape[0]})."
            self.history.add_record(f"Error: {msg}", level="ERROR")
            raise ValueError(msg)

        self.daxis = daxis
        self.history.add_record("Depth axis (daxis) updated.", level="INFO")

    def set_start_time(self, start_time: datetime.datetime) -> None:
        """
        Set the absolute start time.

        Args:
            start_time (datetime.datetime): The start time.

        Raises:
            TypeError: If start_time is not a datetime.datetime object.
        """
        if not isinstance(start_time, datetime.datetime):
            self.history.add_record(f"Error: start_time must be a datetime object, got {type(start_time)}.",
                                    level="ERROR")
            raise TypeError("start_time must be a datetime.datetime object.")
        self.start_time = start_time
        self.history.add_record(f"Start time updated to: {start_time.isoformat()}", level="INFO")

    def set_name(self, name: str) -> None:
        """
        Set the name/identifier for the data.

        Args:
            name (str): The new name.
        """
        if not isinstance(name, str):
            self.history.add_record(f"Error: Name must be a string, got {type(name)}.", level="ERROR")
            raise TypeError("Name must be a string.")
        old_name = self.name
        self.name = name
        self.history.add_record(f"Name changed from '{old_name if old_name else 'Unnamed'}' to '{self.name}'.",
                                level="INFO")

    def set_filename(self, *args: str) -> None:
        """
        Set the filename attribute by joining the provided arguments.
        This method is provided for backward compatibility. Prefer using `set_name`.

        Args:
            args (str): Parts of the filename to be joined.
        """
        new_name = "_".join(map(str, args))
        self.history.add_record(
            f"set_filename called with {args}. Setting name to '{new_name}'. Consider using set_name for clarity.",
            level="WARNING")
        self.set_name(new_name)

    def load_npz(self, filename: str) -> None:
        """
        Load data from an .npz file.
        Expected keys: 'data', 'taxis', 'daxis', 'start_time'.

        Args:
            filename (str): Path to the .npz file.

        Raises:
            FileNotFoundError: If the file does not exist.
            KeyError: If required keys are missing in the .npz file.
            ValueError: If data dimensions mismatch or start_time format is invalid.
        """
        if not filename.endswith(".npz"):
            filename_ext = filename + ".npz"
        else:
            filename_ext = filename

        self.history.add_record(f"Attempting to load data from {filename_ext}.", level="INFO")
        try:
            data_structure = np.load(filename_ext, allow_pickle=True)
        except FileNotFoundError:
            self.history.add_record(f"Error: File not found at {filename_ext}", level="ERROR")
            raise

        try:
            loaded_data = data_structure['data']
            loaded_taxis = data_structure['taxis']
            loaded_daxis = data_structure['daxis']
            start_time_raw = data_structure['start_time']
        except KeyError as e:
            self.history.add_record(f"Error: Missing key {e} in NPZ file {filename_ext}", level="ERROR")
            raise KeyError(
                f"NPZ file {filename_ext} is missing key: {e}. Expected 'data', 'taxis', 'daxis', 'start_time'.") from e

        if loaded_data.ndim != 2:
            raise ValueError(f"Loaded 'data' must be 2D, got {loaded_data.ndim}D.")
        if loaded_taxis.ndim != 1:
            raise ValueError(f"Loaded 'taxis' must be 1D, got {loaded_taxis.ndim}D.")
        if loaded_daxis.ndim != 1:
            raise ValueError(f"Loaded 'daxis' must be 1D, got {loaded_daxis.ndim}D.")

        if loaded_data.shape[1] != loaded_taxis.shape[0]:
            raise ValueError(
                f"Loaded data time dimension ({loaded_data.shape[1]}) mismatch with taxis length ({loaded_taxis.shape[0]}).")
        if loaded_data.shape[0] != loaded_daxis.shape[0]:
            raise ValueError(
                f"Loaded data depth dimension ({loaded_data.shape[0]}) mismatch with daxis length ({loaded_daxis.shape[0]}).")

        # Force convert to float for compatibility
        self.data = loaded_data.astype(float)
        self.taxis = loaded_taxis.astype(float)
        self.daxis = loaded_daxis.astype(float)
        self.history.add_record("Converted data, taxis, and daxis to float type.", level="INFO")

        # Handle start_time
        if isinstance(start_time_raw, np.ndarray) and start_time_raw.size == 1:
            start_time_raw = start_time_raw.item()

        if isinstance(start_time_raw, np.datetime64):
            self.start_time = start_time_raw.astype('datetime64[ms]').astype(datetime.datetime)
        elif isinstance(start_time_raw, str):
            try:
                self.start_time = datetime.datetime.fromisoformat(start_time_raw)
            except ValueError:
                try:
                    self.start_time = datetime.datetime.strptime(start_time_raw, '%Y-%m-%d %H:%M:%S.%f')
                except ValueError:
                    try:
                        self.start_time = datetime.datetime.strptime(start_time_raw, '%Y-%m-%d %H:%M:%S')
                    except ValueError as ve_parse:
                        self.history.add_record(
                            f"Error parsing start_time string '{start_time_raw}' from {filename_ext}", level="ERROR")
                        raise ValueError(f"Could not parse start_time string: {start_time_raw}") from ve_parse
        elif isinstance(start_time_raw, datetime.datetime):
            self.start_time = start_time_raw
        else:
            self.history.add_record(f"Error: Unexpected type for start_time ({type(start_time_raw)}) in {filename_ext}",
                                    level="ERROR")
            raise ValueError(f"Unsupported start_time type: {type(start_time_raw)}")

        self.set_name(os.path.basename(filename_ext))  # Use set_name to also log the change
        self.history.add_record(f"Successfully loaded data from {filename_ext}.", level="INFO")

    def savez(self, filename: str) -> None:
        """
        Save the current data, taxis, daxis, and start_time to a standard .npz file.

        Args:
            filename (str): The path to the .npz file where data will be saved.

        Raises:
            ValueError: If data, taxis, or daxis is not set.
        """
        if self.data is None or self.taxis is None or self.daxis is None:
            self.history.add_record("Error: Cannot save, essential data attributes are not set.", level="ERROR")
            raise ValueError("Data, taxis, and daxis must be set before saving.")

        if not filename.endswith('.npz'):
            filename += '.npz'

        np.savez(
            filename,
            data=self.data,
            taxis=self.taxis,
            daxis=self.daxis,
            start_time=self.start_time
        )
        self.history.add_record(f"Data successfully saved to {filename}.", level="INFO")

    def get_info_str(self) -> str:
        """
        Get a summary string of the Data2D object's attributes.
        For array attributes (taxis, daxis), it shows up to the first 10 elements.
        """
        info_lines = [f"--- Data2D Object Summary: {self.name or 'Unnamed'} ---"]

        info_lines.append(f"Name: {self.name if self.name else 'Not set'}")
        info_lines.append(f"Start Time: {self.start_time.isoformat() if self.start_time else 'Not set'}")

        if self.data is not None:
            info_lines.append(f"Data Shape: {self.data.shape}")
        else:
            info_lines.append("Data: Not set")

        # Time Axis
        if self.taxis is not None:
            info_lines.append(f"Time Axis (taxis): Length={self.taxis.shape[0]}")
            if self.taxis.size > 0:
                if self.taxis.size < 10:
                    info_lines.append(f"  Values: {self.taxis}")
                else:
                    info_lines.append(f"  Values (first 10): {self.taxis[:10]}...")
        else:
            info_lines.append("Time Axis (taxis): Not set")

        # Depth Axis
        if self.daxis is not None:
            info_lines.append(f"Depth Axis (daxis): Length={self.daxis.shape[0]}")
            if self.daxis.size > 0:
                if self.daxis.size < 10:
                    info_lines.append(f"  Values: {self.daxis}")
                else:
                    info_lines.append(f"  Values (first 10): {self.daxis[:10]}...")
        else:
            info_lines.append("Depth Axis (daxis): Not set")

        info_lines.append(f"History contains {len(self.history.records)} records.")
        info_lines.append("----------------------------------------------------")
        return "\n".join(info_lines)

    def print_info(self) -> None:
        """
        Print a summary of the Data2D object's attributes.
        """
        print(self.get_info_str())

    def __str__(self) -> str:
        """Return the summary string of the Data2D object."""
        return self.get_info_str()


    # --- Data Manipulation Methods ---
    def select_time(self, start: Union[datetime.datetime, float, int],
                    end: Union[datetime.datetime, float, int]) -> None:
        """
        Crop data along the time axis. Modifies data, taxis, and start_time in place.
        The taxis will be adjusted to start from 0 relative to the new start_time.

        Args:
            start (Union[datetime.datetime, float, int]): Start time for cropping.
                                                         If float/int, interpreted as seconds relative to current start_time.
            end (Union[datetime.datetime, float, int]): End time for cropping.
                                                       If float/int, interpreted as seconds relative to current start_time.
        Raises:
            ValueError: If attributes are not set, or if start > end.
            TypeError: If start/end types are invalid.
        """
        if self.start_time is None or self.taxis is None or self.data is None:
            msg = "Cannot select time: start_time, taxis, or data is not set."
            self.history.add_record(f"Error: {msg}", level="ERROR")
            raise ValueError(msg)

        if isinstance(start, int): start = float(start)
        if isinstance(end, int): end = float(end)

        start_seconds_rel: float
        end_seconds_rel: float

        if isinstance(start, datetime.datetime) and isinstance(end, datetime.datetime):
            if start > end:
                raise ValueError("Start datetime must be less than or equal to end datetime.")
            start_seconds_rel = (start - self.start_time).total_seconds()
            end_seconds_rel = (end - self.start_time).total_seconds()
        elif isinstance(start, float) and isinstance(end, float):
            if start > end:
                raise ValueError("Start time (seconds) must be less than or equal to end time (seconds).")
            start_seconds_rel = start
            end_seconds_rel = end
        else:
            raise TypeError("Start and end must be both datetime objects or both float/int (seconds).")

        time_mask = (self.taxis >= start_seconds_rel) & (self.taxis <= end_seconds_rel)

        if not np.any(time_mask):
            self.history.add_record(
                f"Warning: No data within selected time range [{start_seconds_rel:.2f}s, {end_seconds_rel:.2f}s]. Data is now empty.",
                level="WARNING")
            # Ensure data maintains 2D shape even if empty along time axis
            if self.data.ndim == 2:
                self.data = np.empty((self.data.shape[0], 0), dtype=self.data.dtype)
            else:  # Should not happen if data was properly set
                self.data = np.array([[]]).T
            self.taxis = np.array([])
            self.start_time += datetime.timedelta(seconds=start_seconds_rel)
            return

        actual_crop_start_offset_seconds = float(self.taxis[time_mask][0])

        self.data = self.data[:, time_mask]
        new_taxis = self.taxis[time_mask] - actual_crop_start_offset_seconds
        if isinstance(actual_crop_start_offset_seconds, float) and not np.issubdtype(new_taxis.dtype, np.floating):
            self.taxis = new_taxis.astype(float)
        else:
            self.taxis = new_taxis

        self.start_time += datetime.timedelta(seconds=actual_crop_start_offset_seconds)

        self.history.add_record(
            f"Time range selected. Original window: [{start_seconds_rel:.2f}s, {end_seconds_rel:.2f}s]. New start time: {self.start_time.isoformat()}",
            level="INFO")

    def select_depth(self, start_depth: Union[float, int], end_depth: Union[float, int]) -> None:
        """
        Crop data along the depth axis. Modifies data and daxis in place.
        Note: This does not normalize daxis to start from 0.

        Args:
            start_depth (Union[float, int]): Start depth for cropping.
            end_depth (Union[float, int]): End depth for cropping.

        Raises:
            ValueError: If daxis/data not set, or if start_depth > end_depth.
            TypeError: If start/end_depth types are invalid.
        """
        if self.daxis is None or self.data is None:
            msg = "Cannot select depth: daxis or data is not set."
            self.history.add_record(f"Error: {msg}", level="ERROR")
            raise ValueError(msg)

        if not (isinstance(start_depth, (int, float)) and isinstance(end_depth, (int, float))):
            raise TypeError("Start and end depth must be float or int types.")
        if start_depth > end_depth:
            raise ValueError("Start depth must be less than or equal to end depth.")

        current_daxis = np.array(self.daxis)

        depth_mask = (current_daxis >= start_depth) & (current_daxis <= end_depth)

        if not np.any(depth_mask):
            self.history.add_record(
                f"Warning: No data within selected depth range [{start_depth}, {end_depth}]. Data is now empty.",
                level="WARNING")
            if self.data.ndim == 2:
                self.data = np.empty((0, self.data.shape[1]), dtype=self.data.dtype)
            else:  # Should not happen
                self.data = np.array([[]])
            self.daxis = np.array([])
            return

        self.data = self.data[depth_mask, :]
        self.daxis = current_daxis[depth_mask]
        self.history.add_record(
            f"Depth range selected: [{start_depth}, {end_depth}]. New daxis length: {len(self.daxis)}", level="INFO")

    def shift(self, shift_amount: Union[datetime.timedelta, float, int]) -> None:
        """
        Apply a time shift to the data by adjusting start_time.

        Args:
            shift_amount (Union[datetime.timedelta, float, int]):
                The time shift. If float/int, interpreted as seconds.
        """
        if self.start_time is None:
            self.history.add_record("Error: Cannot shift, start_time is not set.", level="ERROR")
            raise ValueError("start_time is not set. Load data first.")

        shift_delta: datetime.timedelta
        if isinstance(shift_amount, datetime.timedelta):
            shift_delta = shift_amount
        elif isinstance(shift_amount, (int, float)):
            shift_delta = datetime.timedelta(seconds=float(shift_amount))
        else:
            self.history.add_record(f"Error: Invalid type for shift_amount ({type(shift_amount)})", level="ERROR")
            raise TypeError("Shift amount must be timedelta or float/int (seconds).")

        self.start_time += shift_delta
        self.history.add_record(
            f"Time shifted by {shift_delta.total_seconds():.3f}s. New start_time: {self.start_time.isoformat()}",
            level="INFO")

    def right_merge(self, other_data: 'Data2D') -> None:
        """
        Merge another Data2D instance to the right (chronologically).
        Assumes daxis are compatible and other_data starts after self ends.

        Args:
            other_data (Data2D): The Data2D instance to merge.

        Raises:
            TypeError: If other_data is not Data2D.
            ValueError: If attributes are missing, daxis mismatch, or time overlap.
        """
        if not isinstance(other_data, Data2D):
            raise TypeError("other_data must be an instance of Data2D.")

        required_attrs_self = [self.data, self.taxis, self.daxis, self.start_time]
        required_attrs_other = [other_data.data, other_data.taxis, other_data.daxis, other_data.start_time]

        if any(attr is None for attr in required_attrs_self) or \
                any(attr is None for attr in required_attrs_other):
            self.history.add_record(
                "Error: Merge failed, one or more required attributes (data, taxis, daxis, start_time) are None.",
                level="ERROR")
            raise ValueError("All data, taxis, daxis, and start_time must be set in both objects for merging.")

        self_daxis = cast(np.ndarray, self.daxis)
        other_daxis = cast(np.ndarray, other_data.daxis)
        self_taxis = cast(np.ndarray, self.taxis)
        other_taxis = cast(np.ndarray, other_data.taxis)
        self_start_time = cast(datetime.datetime, self.start_time)
        other_start_time = cast(datetime.datetime, other_data.start_time)
        self_data = cast(np.ndarray, self.data)
        other_data_arr = cast(np.ndarray, other_data.data)

        if not np.array_equal(self_daxis, other_daxis):
            self.history.add_record("Error: Depth axes (daxis) do not match. Merge not possible.", level="ERROR")
            raise ValueError("Depth axes (daxis) do not match, merging not possible.")

        if self_taxis.size == 0:
            self.history.add_record(f"Current data is empty. Copying data from '{other_data.name}'.", level="INFO")
            self.data = deepcopy(other_data_arr)
            self.taxis = deepcopy(other_taxis)
            self.start_time = deepcopy(other_start_time)
            if self_daxis.size == 0: self.daxis = deepcopy(other_daxis)
            return

        if other_taxis.size == 0:
            self.history.add_record(f"'{other_data.name}' is empty. Nothing to merge.", level="INFO")
            return

        end_time_self = self_start_time + datetime.timedelta(seconds=float(self_taxis[-1]))

        if other_start_time < end_time_self:
            msg = (f"Cannot merge: start_time of new data ({other_start_time.isoformat()}) "
                   f"is earlier than or overlaps with the end time of existing data ({end_time_self.isoformat()}).")
            self.history.add_record(f"Error: {msg}", level="ERROR")
            raise ValueError(msg)

        time_offset_seconds = (other_start_time - self_start_time).total_seconds()
        taxis_shifted_other = other_taxis + time_offset_seconds

        if not np.issubdtype(self_taxis.dtype, np.floating) and isinstance(time_offset_seconds, float):
            self_taxis_final = self_taxis.astype(float)
        else:
            self_taxis_final = self_taxis

        if not np.issubdtype(taxis_shifted_other.dtype, np.floating) and isinstance(time_offset_seconds, float):
            taxis_shifted_other_final = taxis_shifted_other.astype(float)
        else:
            taxis_shifted_other_final = taxis_shifted_other

        if np.issubdtype(self_taxis_final.dtype, np.floating) != np.issubdtype(taxis_shifted_other_final.dtype,
                                                                               np.floating):
            self_taxis_final = self_taxis_final.astype(float)
            taxis_shifted_other_final = taxis_shifted_other_final.astype(float)

        self.taxis = np.concatenate((self_taxis_final, taxis_shifted_other_final))
        self.data = np.concatenate((self_data, other_data_arr), axis=1)

        self.history.add_record(
            f"Successfully merged with '{other_data.name}'. New time axis length: {len(self.taxis)}", level="INFO")

    # --- Copy Methods ---
    def __copy__(self) -> 'Data2D':
        """Shallow copy."""
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        result.history = InfoManagementSystem()
        result.history.records = list(self.history.records)
        result.history.add_record(f"Shallow copied from '{self.name if self.name else 'Unnamed Data2D'}'",
                                  level="SYSTEM")
        return result

    def copy(self) -> 'Data2D':
        """Deep copy."""
        new_copy = deepcopy(self)
        new_copy.history = InfoManagementSystem()
        for record in self.history.records:
            new_copy.history.records.append(deepcopy(record))
        new_copy.history.add_record(f"Deep copied from '{self.name if self.name else 'Unnamed Data2D'}'",
                                    level="SYSTEM")
        return new_copy

    # --- History/Log Compatibility ---
    def record_log(self, *args: Any, level: str = "INFO") -> None:
        """
        Record a log entry. Provided for backward compatibility.
        Prefer using self.history.add_record().

        Args:
            *args: Message components to be logged.
            level (str): Severity level of the log entry.
        """
        description = " ".join(map(str, args))
        self.history.add_record(description, level=level)

    def print_log(self, filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None) -> None:
        """
        Prints all recorded history log entries.
        Provided for backward compatibility. Prefer using `print(self.history)`.

        Args:
            filter_fn (callable, optional): A function to filter records.
        """
        self.history.print_records(filter_fn=filter_fn)

    # --- Plotting ---
    def plot(self, ax: Optional[Axes] = None, method: str = 'imshow',
             use_timestamp: bool = False, xaxis_rotation: float = 0,
             xtick_n: int = 5, ytick_n: Optional[int] = None,
             **kwargs: Any) -> Union[AxesImage, QuadMesh, None]:
        """
        Plot the 2D data.

        Args:
            ax (Optional[Axes]): Matplotlib axes to plot on. If None, new figure/axes created.
            method (str): Plotting method: 'imshow' or 'pcolormesh'. Default 'imshow'.
            use_timestamp (bool): If True, use absolute datetime for x-axis. Default False (seconds).
                                  This can be overridden by 'useTimeStamp' in kwargs for compatibility.
            xaxis_rotation (float): Rotation angle for x-axis tick labels. Default 0.
            xtick_n (int): Approximate number of x-axis ticks. Default 5.
            ytick_n (Optional[int]): Approximate number of y-axis ticks. Default None (auto).
            **kwargs: Additional arguments for the plotting function (e.g., cmap, vmin, vmax).
                      Note: 'aspect' is handled separately and applied to the axes.
                      Can also include 'useTimeStamp' (camelCase) for backward compatibility.

        Returns:
            Union[AxesImage, QuadMesh, None]: The Matplotlib artist object, or None if data is empty.
        """
        if self.data is None or self.taxis is None or self.daxis is None:
            self.history.add_record("Error: Cannot plot. Data, taxis, or daxis is not set.", level="ERROR")
            raise ValueError("Data, taxis, or daxis is not set.")
        if self.data.size == 0 or self.taxis.size == 0 or self.daxis.size == 0:
            self.history.add_record("Warning: Attempting to plot empty data. Plot may be blank.", level="WARNING")
            if ax is None:
                fig, ax_new = plt.subplots()
                ax_new.set_title(kwargs.get('title', "Empty Data"))
                ax_new.set_xlabel("Time")
                ax_new.set_ylabel("Depth/Spatial")
                if not plt.isinteractive(): plt.show()
            return None

        # Handle plot-specific kwargs and determine effective use_timestamp
        # Start with the formal argument's value for use_timestamp
        effective_use_timestamp = use_timestamp
        if 'useTimeStamp' in kwargs:  # Legacy camelCase in kwargs takes precedence
            effective_use_timestamp = bool(kwargs.pop('useTimeStamp'))
        elif 'use_timestamp' in kwargs:  # snake_case in kwargs (if passed explicitly)
            effective_use_timestamp = bool(kwargs.pop('use_timestamp'))

        # Pop other plot-specific parameters from kwargs
        aspect_to_set = kwargs.pop('aspect', 'auto')
        figsize = kwargs.pop('figsize', (8, 6))
        plot_title_kwarg = kwargs.pop('title', None)
        time_format_kwarg = kwargs.pop('time_format', '%H:%M:%S')
        ylabel_kwarg = kwargs.pop('ylabel', 'Depth / Spatial Axis')
        colorbar_kwarg = kwargs.pop('colorbar', False)
        clabel_kwarg = kwargs.pop('clabel', None)

        # Pop formal arguments if they were also passed via kwargs (less likely but good practice)
        kwargs.pop('xaxis_rotation', None)
        kwargs.pop('xtick_n', None)
        kwargs.pop('ytick_n', None)
        # Pop legacy versions of tickN if they exist from old test calls
        kwargs.pop('xtickN', None)
        kwargs.pop('ytickN', None)

        plot_taxis_to_use: np.ndarray
        if effective_use_timestamp:
            if self.start_time is None:
                self.history.add_record("Error: Timestamp plotting requested but start_time is not set.", level="ERROR")
                raise ValueError("start_time must be set to use timestamps for plotting.")
            plot_taxis_to_use = self.calculate_time()
            x_label = 'Time (Absolute)'
        else:
            plot_taxis_to_use = self.taxis
            x_label = f'Time (seconds since {self.start_time.strftime("%Y-%m-%d %H:%M:%S") if self.start_time else "start"})'

        plot_daxis_to_use = self.daxis

        new_figure_created = False
        current_fig = None
        if ax is None:
            current_fig, ax = plt.subplots(figsize=figsize)
            new_figure_created = True
        else:
            current_fig = ax.get_figure()

        sort_idx_d = np.argsort(plot_daxis_to_use)
        sorted_daxis = plot_daxis_to_use[sort_idx_d]
        if self.data.ndim == 2:
            sorted_data = self.data[sort_idx_d, :]
        else:
            self.history.add_record(
                f"Warning: Data is not 2D (shape: {self.data.shape}). Plotting may fail or be incorrect.",
                level="WARNING")
            sorted_data = self.data

        img_artist: Union[AxesImage, QuadMesh]
        # Remaining kwargs are now only for imshow/pcolormesh
        if method == 'imshow':
            numeric_plot_taxis_for_extent = mdates.date2num(
                plot_taxis_to_use) if effective_use_timestamp else plot_taxis_to_use
            img_artist = ax.imshow(sorted_data,
                                   extent=[numeric_plot_taxis_for_extent[0], numeric_plot_taxis_for_extent[-1],
                                           sorted_daxis[-1], sorted_daxis[0]],
                                   **kwargs)
        elif method == 'pcolormesh':
            img_artist = ax.pcolormesh(plot_taxis_to_use, sorted_daxis, sorted_data,
                                       shading=kwargs.pop('shading', 'auto'),  # Pop shading specifically for pcolormesh
                                       **kwargs)
            ax.invert_yaxis()
        else:
            self.history.add_record(f"Error: Invalid plot method '{method}'. Choose 'imshow' or 'pcolormesh'.",
                                    level="ERROR")
            raise ValueError(f"Invalid plot method: {method}")

        if aspect_to_set is not None:
            ax.set_aspect(aspect_to_set)

        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel_kwarg)

        if effective_use_timestamp:
            ax.xaxis.set_major_formatter(mdates.DateFormatter(time_format_kwarg))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=xtick_n))
            if current_fig: current_fig.autofmt_xdate(rotation=xaxis_rotation)
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=xtick_n))
            ax.tick_params(axis='x', labelrotation=xaxis_rotation)

        if ytick_n is not None:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=ytick_n))

        if plot_title_kwarg is not None:
            ax.set_title(plot_title_kwarg)
        elif new_figure_created and self.name:
            ax.set_title(f"Data: {self.name}")

        if colorbar_kwarg:
            plt.colorbar(img_artist, ax=ax, label=clabel_kwarg)

        if new_figure_created and not plt.isinteractive():
            plt.show()

        self.history.add_record(
            f"Plot generated using '{method}' method for '{self.name if self.name else 'Unnamed Data2D'}'.",
            level="INFO")
        return img_artist

    # --- Utility & Calculation Methods ---
    def get_start_time(self) -> Optional[datetime.datetime]:
        """Returns the absolute start time of the data."""
        return self.start_time

    def get_end_time(self, time_format: str = 'datetime') -> Union[Optional[datetime.datetime], Optional[float]]:
        """
        Calculate and return the absolute end time of the data.

        Args:
            time_format (str): Desired return type: 'datetime' or 'seconds' (relative to start_time).
                               Default 'datetime'.
        Returns:
            Union[Optional[datetime.datetime], Optional[float]]: End time, or None if not determinable.
        """
        if self.start_time is None or self.taxis is None or self.taxis.size == 0:
            self.history.add_record("Warning: Cannot get end time, start_time or taxis not set/empty.", level="WARNING")
            return None

        last_taxis_val = float(self.taxis[-1])
        if time_format == 'datetime':
            return self.start_time + datetime.timedelta(seconds=last_taxis_val)
        elif time_format == 'seconds':
            return last_taxis_val
        else:
            self.history.add_record(f"Error: Invalid type for get_end_time: '{time_format}'.", level="ERROR")
            raise ValueError("time_format for get_end_time must be 'datetime' or 'seconds'.")

    def calculate_time(self) -> np.ndarray:
        """
        Calculate absolute datetime values for the time axis.

        Returns:
            np.ndarray: Array of np.datetime64 objects.
        Raises:
            ValueError: If start_time or taxis is not set.
        """
        if self.start_time is None: raise ValueError("start_time is not set.")
        if self.taxis is None: raise ValueError("taxis is not set.")

        try:
            taxis_for_timedelta = self.taxis
            if not np.issubdtype(taxis_for_timedelta.dtype, np.floating):
                taxis_for_timedelta = self.taxis.astype(float)
            timedeltas = np.array(taxis_for_timedelta, dtype='timedelta64[s]')
            return np.datetime64(self.start_time) + timedeltas
        except Exception as e:
            self.history.add_record(f"Warning: Using fallback for time calculation due to: {e}", level="WARNING")
            return np.array([self.start_time + datetime.timedelta(seconds=float(t)) for t in self.taxis])

    def calculate_time_seconds(self) -> Optional[np.ndarray]:
        """Returns the time axis in seconds relative to start_time."""
        return self.taxis

    def apply_lowpass_filter(self, cutoff_freq: float, sample_rate: Optional[float] = None, order: int = 5) -> None:
        """
        Apply a low-pass filter to the data along the time axis (axis=1).

        Args:
            cutoff_freq (float): Cutoff frequency for the low-pass filter.
            sample_rate (Optional[float]): Sample rate. If None, calculated from taxis.
            order (int): Filter order. Default 5.
        """
        if self.data is None or self.taxis is None:
            raise ValueError("Data and taxis must be set to apply filter.")
        if self.data.size == 0 or self.taxis.size < 2:
            self.history.add_record("Skipped lowpass filter: data empty or taxis too short.", level="WARNING")
            return

        dt: float
        if sample_rate is None:
            dt = np.mean(np.diff(self.taxis))
            if dt <= 0:
                raise ValueError("Calculated dt from taxis is non-positive. Check taxis.")
            actual_sample_rate = 1.0 / dt
        else:
            actual_sample_rate = sample_rate
            dt = 1.0 / actual_sample_rate

        nyquist_freq = 0.5 * actual_sample_rate
        if cutoff_freq >= nyquist_freq:
            self.history.add_record(
                f"Warning: Cutoff frequency ({cutoff_freq}Hz) is >= Nyquist frequency ({nyquist_freq}Hz). Filter might not be effective or stable.",
                level="WARNING")

        filtered_data = np.zeros_like(self.data)
        for i in range(self.data.shape[0]):
            filtered_data[i, :] = signal_utils.bpfilter(
                self.data[i, :],
                dt=dt,
                lowcut=0,
                highcut=cutoff_freq,
                order=order,
                axis=-1
            )
        self.data = filtered_data
        self.history.add_record(
            f"Applied low-pass filter: cutoff={cutoff_freq}Hz, order={order}, sample_rate_used={actual_sample_rate:.2f}Hz.",
            level="INFO")

    def get_value_by_depth(self, depth: float, interpolate: bool = False) -> Optional[np.ndarray]:
        """
        Get the 1D time series data for the channel nearest to the specified depth.

        Args:
            depth (float): The target depth.
            interpolate (bool): If True, and exact depth not found, interpolate between nearest channels.
                                (Currently NOT IMPLEMENTED, finds nearest). Default False.

        Returns:
            Optional[np.ndarray]: 1D array of data for the selected depth channel, or None if invalid.
        """
        if self.daxis is None or self.data is None:
            raise ValueError("daxis and data must be set.")
        if self.daxis.size == 0:
            return None

        if depth < np.min(self.daxis) or depth > np.max(self.daxis):
            self.history.add_record(
                f"Error: Requested depth {depth} is out of daxis range [{np.min(self.daxis)}, {np.max(self.daxis)}].",
                level="ERROR")
            return None

        if interpolate:
            self.history.add_record(
                "Warning: Depth interpolation is not yet implemented in get_value_by_depth. Finding nearest.",
                level="WARNING")
            # Implement in the future. -- Shenyao
            pass

        depth_idx = np.argmin(np.abs(self.daxis - depth))
        self.history.add_record(f"Retrieved data for depth nearest to {depth} (actual: {self.daxis[depth_idx]}).",
                                level="INFO")
        return self.data[depth_idx, :]

    def get_value_by_time(self, time_point: Union[datetime.datetime, float, int]) -> Tuple[np.ndarray, float]:
        """
        Get the 1D depth trace data for the timestamp nearest to the specified time point.

        Args:
            time_point (Union[datetime.datetime, float, int]): The target time.
                If float/int, interpreted as seconds relative to current start_time.
                If datetime.datetime, it's an absolute time.

        Returns:
            Tuple[np.ndarray, float]: A tuple containing:
                - The 1D array of data for the selected time step.
                - The closest time value found in the taxis (in seconds).

        Raises:
            ValueError: If taxis, data, or start_time (for datetime input) is not set.
            TypeError: If time_point is not a supported type.
        """
        if self.taxis is None or self.data is None:
            raise ValueError("taxis and data must be set.")
        if self.taxis.size == 0:
            raise ValueError("Cannot get value by time, taxis is empty.")

        target_seconds: float
        if isinstance(time_point, (int, float)):
            target_seconds = float(time_point)
        elif isinstance(time_point, datetime.datetime):
            if self.start_time is None:
                raise ValueError("start_time must be set to use a datetime object as input.")
            target_seconds = (time_point - self.start_time).total_seconds()
        else:
            raise TypeError(f"Unsupported type for time_point: {type(time_point)}")

        if not (self.taxis.min() <= target_seconds <= self.taxis.max()):
            self.history.add_record(
                f"Warning: Requested time {target_seconds:.2f}s is outside the taxis range "
                f"[{self.taxis.min():.2f}s, {self.taxis.max():.2f}s]. Finding nearest endpoint.",
                level="WARNING")

        time_idx = np.argmin(np.abs(self.taxis - target_seconds))
        closest_time_val = self.taxis[time_idx]

        self.history.add_record(
            f"Retrieved data for time nearest to {target_seconds:.2f}s (actual: {closest_time_val:.2f}s).",
            level="INFO")

        return self.data[:, time_idx], float(closest_time_val)

    def get_max_daxis(self) -> Optional[float]:
        """Returns the maximum value of the depth axis (daxis)."""
        if self.daxis is None or self.daxis.size == 0:
            return None
        return float(self.daxis[-1])

    def get_max_taxis(self) -> Optional[float]:
        """Returns the maximum value of the time axis (taxis) in seconds."""
        if self.taxis is None or self.taxis.size == 0:
            return None
        return float(self.taxis[-1])
