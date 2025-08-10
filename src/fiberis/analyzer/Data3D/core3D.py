# src/fiberis/analyzer/Data3D/core3D.py
# Core module for handling 3D data, with automated discovery and history logging.
# This implementation reads a single variable from a user-selected series of CSVs.
# Shenyao Jin, 2025/06/23, shenyaojin@mines.edu

import os
import re
import glob
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from fiberis.utils.history_utils import InfoManagementSystem


class Data3D:
    """
    Class for handling data from a MOOSE VectorPostprocessor CSV file series.
    It can auto-discover multiple samplers in a directory and load a specific one,
    assembling a data array for a single variable and its x, y axes.

    Attributes:
        data (Optional[np.ndarray]): Data array of the variable's values. Shape: (n_spatial_points, n_time_points).
        taxis (Optional[np.ndarray]): 1D array for the time axis (in seconds).
        xaxis (Optional[np.ndarray]): 1D array for the x-coordinate of each spatial point.
        yaxis (Optional[np.ndarray]): 1D array for the y-coordinate of each spatial point.
        variable_name (Optional[str]): Name of the loaded variable.
        name (Optional[str]): The base name of the loaded sampler, e.g., '..._pressure_profile_bot_hf'.
        history (InfoManagementSystem): An object that logs the history of operations.
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize the Data3D class.
        """
        self.data: Optional[np.ndarray] = None
        self.taxis: Optional[np.ndarray] = None
        self.xaxis: Optional[np.ndarray] = None
        self.yaxis: Optional[np.ndarray] = None
        self.variable_name: Optional[str] = None
        self.name: Optional[str] = name
        self.history: InfoManagementSystem = InfoManagementSystem()
        self.history.add_record("Data3D object initialized.", level="INFO")

    def load_npz(self, filename: str):
        """
        Loads data from a standard .npz file.
        """
        self.history.add_record(f"Loading data from NPZ file: {filename}", level="INFO")
        if not filename.endswith('.npz'):
            filename += '.npz'

        try:
            data_structure = np.load(filename, allow_pickle=True)
            self.data = data_structure['data']
            self.taxis = data_structure['taxis']
            self.xaxis = data_structure['xaxis']
            self.yaxis = data_structure['yaxis']
            self.variable_name = str(data_structure['variable_name'])
            self.name = str(data_structure['name'])
            self.history.add_record(f"Successfully loaded data from {filename}.", level="INFO")
        except FileNotFoundError:
            self.history.add_record(f"Error: File not found at {filename}", level="ERROR")
            raise
        except KeyError as e:
            self.history.add_record(f"Error: Missing key {e} in NPZ file {filename}", level="ERROR")
            raise

    def savez(self, filename: str):
        """
        Saves the current data to a standard .npz file.
        """
        if self.data is None or self.taxis is None or self.xaxis is None or self.yaxis is None:
            self.history.add_record("Error: Cannot save, essential data attributes are not set.", level="ERROR")
            raise ValueError("Data and axes must be set before saving.")

        if not filename.endswith('.npz'):
            filename += '.npz'

        np.savez(
            filename,
            data=self.data,
            taxis=self.taxis,
            xaxis=self.xaxis,
            yaxis=self.yaxis,
            variable_name=self.variable_name,
            name=self.name
        )
        self.history.add_record(f"Data successfully saved to {filename}.", level="INFO")

    