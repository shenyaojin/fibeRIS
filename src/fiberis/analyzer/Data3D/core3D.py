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

    def load_from_csv_series(self,
                             directory: str,
                             post_processor_id: int = 0,
                             variable_index: int = 1):
        """
        Loads data by auto-discovering all vector samplers and the time series file.
        It uses files ending in '_0000.csv' to identify each unique sampler.

        Args:
            directory (str): The directory containing the MOOSE output CSV files.
            post_processor_id (int): The integer index of the sampler to load. Samplers are
                                     sorted alphabetically by their base name. Defaults to 0.
            variable_index (int): The column index of the variable to read from the vector CSVs.
                                  Defaults to 1 (the second column, after 'id').
        """
        self.history.add_record(f"Starting to load data from directory: {directory}", level="INFO")

        # 1. Discover all unique samplers
        all_files = os.listdir(directory)
        sampler_files_0000 = [f for f in all_files if f.endswith('_0000.csv')]

        if not sampler_files_0000:
            msg = f"No samplers found. Could not find files ending in '_0000.csv' in '{directory}'."
            self.history.add_record(msg, level="ERROR")
            raise FileNotFoundError(msg)

        sampler_base_names = sorted([f.replace('_0000.csv', '') for f in sampler_files_0000])
        self.history.add_record(f"Discovered {len(sampler_base_names)} samplers: {sampler_base_names}", level="INFO")

        if not (0 <= post_processor_id < len(sampler_base_names)):
            msg = f"post_processor_id {post_processor_id} is out of range. Select an ID between 0 and {len(sampler_base_names) - 1}."
            self.history.add_record(msg, level="ERROR")
            raise IndexError(msg)

        self.name = sampler_base_names[post_processor_id]
        self.history.add_record(f"Selected sampler ID {post_processor_id}: '{self.name}'", level="INFO")

        # 2. Discover the time series file
        numbered_files = {f for f in all_files if re.match(r'.*?_\d+\.csv$', f)}
        all_csv_files = {f for f in all_files if f.endswith('.csv')}
        time_files = list(all_csv_files - numbered_files)

        if len(time_files) != 1:
            msg = f"Expected 1 time series CSV file, but found {len(time_files)}: {time_files}"
            self.history.add_record(msg, level="ERROR")
            raise FileNotFoundError(msg)

        time_csv_path = os.path.join(directory, time_files[0])
        self.history.add_record(f"Discovered time series file: '{time_files[0]}'", level="INFO")

        # 3. Load time vector and the selected vector file series
        try:
            time_df = pd.read_csv(time_csv_path)
            self.taxis = time_df.iloc[:, 0].to_numpy()
        except Exception as e:
            msg = f"Could not load time vector from '{time_csv_path}'. Error: {e}"
            self.history.add_record(msg, level="ERROR")
            raise ValueError(msg)

        vector_files_pattern = os.path.join(directory, f"{self.name}_*.csv")
        vector_files = sorted(glob.glob(vector_files_pattern))

        if len(vector_files) != len(self.taxis):
            msg = f"Mismatch: {len(vector_files)} vector files found, but {len(self.taxis)} time steps. Truncating to shorter length."
            self.history.add_record(msg, level="WARNING")
            min_len = min(len(vector_files), len(self.taxis))
            vector_files = vector_files[:min_len]
            self.taxis = self.taxis[:min_len]

        # 4. Process the file series
        data_list = []
        spatial_axes_established = False
        for file_path in vector_files:
            try:
                df = pd.read_csv(file_path)
                if df.empty:
                    num_spatial_points = len(self.xaxis) if self.xaxis is not None else 0
                    slice_data = np.full(num_spatial_points, np.nan)
                else:
                    slice_data = df.iloc[:, variable_index].to_numpy()
                    if not spatial_axes_established:
                        self.xaxis = df.iloc[:, -3].to_numpy()
                        self.yaxis = df.iloc[:, -2].to_numpy()
                        self.variable_name = df.columns[variable_index]
                        spatial_axes_established = True
            except Exception as e:
                self.history.add_record(f"Could not process file {file_path}. Error: {e}", level="WARNING")
                num_spatial_points = len(self.xaxis) if self.xaxis is not None else 0
                slice_data = np.full(num_spatial_points, np.nan)
            data_list.append(slice_data)

        # 5. Finalize the data array
        if not spatial_axes_established:
            self.history.add_record("All CSV files for this sampler were empty. Data object is empty.", level="WARNING")
            self.data = np.empty((len(self.taxis), 0))
        else:
            full_len = len(self.xaxis)
            for i, arr in enumerate(data_list):
                if len(arr) != full_len:
                    data_list[i] = np.full(full_len, np.nan)

            # Stack to create (time, space) array
            self.data = np.stack(data_list, axis=0)

            # Transpose the data to match the expected shape (space, time)
            self.data = self.data.T
            self.history.add_record(f"Successfully loaded and transposed data. Final shape: {self.data.shape}",
                                    level="INFO")