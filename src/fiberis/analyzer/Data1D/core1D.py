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

    def shift(self, shift):
        """
        Apply a time shift to the data.

        Parameters:
        ----------
        shift : timedelta or float
            The time shift to apply.
        """
        if isinstance(shift, datetime.timedelta):
            shift_seconds = shift.total_seconds()
        elif isinstance(shift, (int, float)):
            shift_seconds = shift
        else:
            raise TypeError("Shift must be either a timedelta or a float representing seconds.")

        self.taxis = self.taxis + shift_seconds

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

    def plot(self, ax=None, start=None, end=None, title=None, **kwargs):
        """
        Plot the data on a specified axis within a time range.

        Parameters:
        ----------
        ax : matplotlib.axes.Axes, optional
            The axis to plot on. If None, a new figure and axis are created.
        start : float or datetime.datetime, optional
            The start time for the plot. If None, plotting starts from the beginning.
        end : float or datetime.datetime, optional
            The end time for the plot. If None, plotting ends at the last data point.
        title : str or None, optional
            The title for the plot. If None, no title is set. If not provided, defaults to the instance's name.
        kwargs : dict
            Additional keyword arguments to pass to the `plot` function.
        """
        if start is not None and end is not None:
            self.crop(start, end)

        if self.start_time is not None:
            time_axis = self.calculate_time()
        else:
            time_axis = self.taxis

        # Create a new figure and axis if ax is not provided
        new_figure_created = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            new_figure_created = True

        ax.plot(time_axis, self.data, label=self.name or 'Data', **kwargs)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')

        # Set the title based on the input parameter
        if title is not None:
            ax.set_title(title)
        elif self.name:
            ax.set_title(self.name)

        # Ensure the legend updates properly by adding a check for existing handles
        handles, labels = ax.get_legend_handles_labels()
        if not handles or self.name not in labels:
            ax.legend()

        ax.grid(True)

        # Only show the plot if a new figure was created
        if new_figure_created:
            plt.show()
