# Define the class Data1D and its methods
# Shenyao Jin, shenyaojin@mines.edu, 02/06/2025

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from copy import deepcopy


class Data1D:
    """
    Class for handling one-dimensional data.

    Attributes:
        taxis (NoneType): Represents the time axis, initialized to None.
        data (NoneType): Represents the data values, initialized to None.
        start_time (NoneType): Represents the start time, initialized to None.
    """

    def __init__(self):
        """
        Initialize the Data1D class with default values.

        taxis (NoneType): Initialized to None.
        data (NoneType): Initialized to None.
        start_time (NoneType): Initialized to None.
        """
        self.taxis = None
        self.data = None
        self.start_time = None

        self.name = None

    def load_npz(self, filename):
        """
        Load data from an npz file and set the class attributes.

        :param filename: The filename of the npz file.
        :return: None
        """
        if filename[-4:] != '.npz':
            filename += '.npz'

        data_structure = np.load(filename, allow_pickle=True)
        self.data = data_structure['data']
        self.taxis = data_structure['taxis']
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

        self.name = filename

    def crop(self, start, end):
        """
        Crop the data to a specific time range.

        Parameters:
        ----------
        start : datetime, float, or int
            The start time for cropping.
        end : datetime, float, or int
            The end time for cropping.
        """
        # Convert int to float for uniformity
        if isinstance(start, int):
            start = float(start)
        if isinstance(end, int):
            end = float(end)

        # Validate input types
        if not ((isinstance(start, (datetime.datetime, float))) and (isinstance(end, (datetime.datetime, float)))):
            raise TypeError("Start and end must be either datetime, float, or int types.")

        # Ensure start is less than or equal to end
        if start > end:
            raise ValueError("Start time must be less than or equal to end time.")

        if isinstance(start, datetime.datetime) and isinstance(end, datetime.datetime):
            # Convert datetime range to seconds if needed
            start_seconds = (start - self.start_time).total_seconds()
            end_seconds = (end - self.start_time).total_seconds()
        elif isinstance(start, float) and isinstance(end, float):
            start_seconds = start
            end_seconds = end
        else:
            raise ValueError("Start and end must be of the same type.")

        # Crop the data and time axis
        ind = (self.taxis >= start_seconds) & (self.taxis <= end_seconds)
        self.data = self.data[ind]
        self.taxis = self.taxis[ind]
        # Update the start time based on the start of the cropped data
        self.start_time = self.start_time + datetime.timedelta(seconds=start_seconds)
        # shift the time axis to start from 0
        self.taxis -= start_seconds

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

    def get_value_by_time(self, time):
        """
        Get the gauge data value at a specific time.

        Parameters:
        ----------
        time : float
            The time at which to get the data value.

        Note: If you find there's a performance issue, you should crop the data first.
        """
        taxis_sec = self.taxis  # Assuming taxis is already in seconds format
        value = np.interp(time, taxis_sec, self.data)
        return value

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

    def copy(self):
        """
        Create a deep copy of the Data1D instance.

        Returns:
        --------
        Data1D
            A new instance of Data1D with the same data and attributes.
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

    def plot(self, ax=None, title=None, useTimeStamp=False, useLegend=False, **kwargs):

        if useTimeStamp:
            time_axis = self.calculate_time()
        else:
            time_axis = self.taxis

        # Create a new figure and axis if ax is not provided
        new_figure_created = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            new_figure_created = True

        # Ensure that a unique label is always set for the legend
        plot_label = self.name if self.name else 'Data'
        img = ax.plot(time_axis, self.data, label=plot_label, **kwargs)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')

        # Set the title based on the input parameter
        if title is not None:
            ax.set_title(title)

        # Explicitly request the legend to display
        if useLegend:
            ax.legend(loc='best', fontsize='medium')

        # Only show the plot if a new figure was created
        if new_figure_created:
            plt.show()

        return img

    def right_merge(self, data):
        """
        Merge another Data1D instance to the right along the time axis.

        Parameters:
        ----------
        data : Data1D (or subclass)
            The Data1D-compatible instance to merge with the current instance.

        Raises:
        -------
        ValueError
            If the start_time of `data` is earlier than the end time of `self`.

        """
        # Calculate end time
        end_time_self = self.start_time + datetime.timedelta(seconds=float(self.taxis[-1]))

        if data.start_time < end_time_self:
            raise ValueError(
                f"Cannot merge: start_time of new data ({data.start_time}) "
                f"is earlier than the end time of existing data ({end_time_self})."
            )

        # Calculate the shifted time
        taxis_shifted = data.taxis + (data.start_time - self.start_time).total_seconds()

        # Concatenate the time axis
        self.taxis = np.concatenate((self.taxis, taxis_shifted))
        self.data = np.concatenate((self.data, data.data))

        # Logging the message
        print(f"Merged with {data.name} at time {data.start_time}")