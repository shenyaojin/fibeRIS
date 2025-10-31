# Core 1D for geometric data structures, like those "locations" in 1D space.
# Shenyao Jin, 09/17/2025

import numpy as np
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from fiberis.utils import history_utils
from typing import Optional, Union, List, Any
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

class Data1DG:
    """
    Class for handling 1D geometric/spatial data.

    This class is structurally analogous to the Data1D class but replaces the time axis
    with a spatial axis (e.g., depth, distance). It is designed to hold data values
    at specific locations along a 1D path.

    Attributes:
        daxis (Optional[np.ndarray]): The spatial axis (e.g., depth), analogous to 'taxis' in Data1D.
        data (Optional[np.ndarray]): Data values corresponding to the daxis.
        axis_name (str): The name and unit of the spatial axis (e.g., 'Measured Depth (ft)').
        name (Optional[str]): Name or identifier for the data.
        history (history_utils.InfoManagementSystem): System for logging operations.
    """

    def __init__(self, data: Optional[np.ndarray] = None,
                 daxis: Optional[np.ndarray] = None,
                 axis_name: str = 'Spatial Axis',
                 name: Optional[str] = None):
        """
        Initialize the Data1DG class.

        Args:
            data (Optional[np.ndarray]): Initial data array.
            daxis (Optional[np.ndarray]): Initial spatial axis.
            axis_name (str): Name for the spatial axis.
            name (Optional[str]): Initial name for the data.
        """
        self.daxis: Optional[np.ndarray] = np.array(daxis, dtype=float) if daxis is not None else None
        self.data: Optional[np.ndarray] = np.array(data, dtype=float) if data is not None else None
        self.axis_name: str = axis_name
        self.name: Optional[str] = name
        self.history: history_utils.InfoManagementSystem = history_utils.InfoManagementSystem()

        if self.name:
            self.history.add_record(f"Initialized Data1DG object with name: {self.name}")
        else:
            self.history.add_record("Initialized empty Data1DG object.")

        if self.daxis is not None and self.data is not None:
            if len(self.daxis) != len(self.data):
                self.history.add_record("Error: daxis and data must have the same length.", level="ERROR")
                raise ValueError("daxis and data must have the same length.")

    def load_npz(self, filename: str) -> None:
        """
        Load data from an .npz file.
        The .npz file is expected to contain 'daxis', 'data', and optionally 'axis_name'.

        Args:
            filename (str): The path to the .npz file.
        """
        self.history.add_record(f"Attempting to load data from NPZ file: {filename}", level="INFO")
        if not os.path.exists(filename):
            self.history.add_record(f"Error: File not found at {filename}", level="ERROR")
            raise FileNotFoundError(f"The specified NPZ file does not exist: {filename}")

        with np.load(filename, allow_pickle=True) as data_structure:
            self.daxis = data_structure['daxis'].astype(float)
            self.data = data_structure['data'].astype(float)
            if 'axis_name' in data_structure:
                self.axis_name = str(data_structure['axis_name'].item())

        self.name = os.path.basename(filename)
        self.history.add_record(f"Successfully loaded data from {filename}. Name set to '{self.name}'.", level="INFO")

    def savez(self, filename: Optional[str] = None) -> None:
        """
        Save the current data to an .npz file.

        Args:
            filename (Optional[str]): The path to the .npz file. If None, self.name is used.
        """
        if self.daxis is None or self.data is None:
            raise ValueError("Cannot save, daxis or data attribute is not set.")

        save_filename = filename if filename else self.name
        if not save_filename:
            raise ValueError("Filename must be provided if object name is not set.")

        if not save_filename.lower().endswith('.npz'):
            save_filename += '.npz'

        np.savez(save_filename,
                 daxis=self.daxis,
                 data=self.data,
                 axis_name=self.axis_name)
        self.history.add_record(f"Data saved to {save_filename}.", level="INFO")

    def select_range(self, start_loc: float, end_loc: float) -> None:
        """
        Crop the data to a specific range of locations. Modifies data in place.

        Args:
            start_loc (float): The start of the range.
            end_loc (float): The end of the range.
        """
        if self.daxis is None:
            raise ValueError("Cannot select range, daxis is not set.")
        if start_loc > end_loc:
            raise ValueError("start_loc must be less than or equal to end_loc.")

        mask = (self.daxis >= start_loc) & (self.daxis <= end_loc)
        self.daxis = self.daxis[mask]
        if self.data is not None:
            self.data = self.data[mask]

        self.history.add_record(f"Data cropped to range [{start_loc}, {end_loc}].", level="INFO")

    def shift(self, offset: float) -> None:
        """
        Apply a spatial shift to the daxis.

        Args:
            offset (float): The value to add to all locations on the daxis.
        """
        if self.daxis is None:
            raise ValueError("Cannot shift, daxis is not set.")

        self.daxis += offset
        self.history.add_record(f"Shifted daxis by {offset}.", level="INFO")

    def get_value_by_location(self, location: float) -> float:
        """
        Get the interpolated data value at a specific location.

        Args:
            location (float): The location at which to get the data value.

        Returns:
            float: The interpolated data value.
        """
        if self.data is None or self.daxis is None or self.data.size == 0:
            raise ValueError("Data or daxis is not loaded or is empty.")

        value = np.interp(location, self.daxis, self.data)
        return float(value)

    def copy(self) -> 'Data1DG':
        """
        Create a deep copy of the Data1DG instance.
        """
        new_copy = deepcopy(self)
        new_copy.history.add_record(f"Created a deep copy from '{self.name if self.name else 'Unnamed Data1DG'}'.")
        return new_copy

    def get_info_str(self) -> str:
        """
        Get a summary string of the Data1DG object's attributes.
        """
        info_lines = [f"--- Data1DG Object Summary: {self.name or 'Unnamed'} ---"]
        info_lines.append(f"Name: {self.name if self.name else 'Not set'}")
        info_lines.append(f"Axis Name: {self.axis_name}")

        if self.daxis is not None:
            info_lines.append(f"Spatial Axis (daxis): Count={len(self.daxis)}, Min={np.min(self.daxis):.2f}, Max={np.max(self.daxis):.2f}")
        else:
            info_lines.append("Spatial Axis (daxis): Not set")

        if self.data is not None:
            info_lines.append(f"Data: Count={len(self.data)}, Min={np.min(self.data):.2f}, Max={np.max(self.data):.2f}")
        else:
            info_lines.append("Data: Not set")

        info_lines.append(f"History contains {len(self.history.records)} records.")
        info_lines.append("----------------------------------------------------")
        return "\n".join(info_lines)

    def print_info(self) -> None:
        """
        Print a summary of the Data1DG object's attributes.
        """
        print(self.get_info_str())

    def __str__(self) -> str:
        """Return the summary string of the Data1DG object."""
        return self.get_info_str()

    def plot(self, ax: Optional[Axes] = None, title: Optional[str] = None, **kwargs: Any) -> List[Line2D]:
        """
        Plot the spatial data.

        Args:
            ax (Optional[plt.Axes]): Matplotlib axes object to plot on. If None, a new figure and axes are created.
            title (Optional[str]): Title for the plot.
            **kwargs: Additional keyword arguments to pass to `ax.plot()`.

        Returns:
            A list containing the Line2D artists added to the axes.
        """
        if self.data is None or self.daxis is None:
            raise ValueError("Cannot plot, data or daxis is not loaded.")

        new_figure_created = False
        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.pop('figsize', (10, 6)))
            new_figure_created = True

        plot_label = self.name if self.name else 'Data'
        if 'label' not in kwargs:
            kwargs['label'] = plot_label

        lines = ax.plot(self.daxis, self.data, **kwargs)

        ax.set_xlabel(self.axis_name)
        ax.set_ylabel('Value')
        plot_title = title if title else (self.name if self.name else "1D Spatial Data")
        ax.set_title(plot_title)
        ax.legend()
        ax.grid(True)

        if new_figure_created:
            plt.tight_layout()
            plt.show()

        self.history.add_record(f"Plot generated for '{self.name if self.name else 'Unnamed Data1DG'}'.", level="INFO")
        return lines