# src/fiberis/io/reader_moose_vpp.py
# Reader for MOOSE VectorPostProcessor CSV file series.
# Shenyao Jin, shenyaojin@mines.edu, 08/09/2025
import datetime
import os
import re
import glob
import numpy as np
import pandas as pd
from fiberis.io import core
from typing import Optional

class MOOSEVectorPostProcessorReader(core.DataIO):
    """
    A reader for handling a series of MOOSE VectorPostprocessor CSV files.
    It auto-discovers samplers, reads the data for a selected variable,
    and converts it to the standard fibeRIS npz format.
    """

    def __init__(self):
        """
        Initialize the MOOSEVectorPostProcessorReader class.
        """
        super().__init__()
        self.xaxis: Optional[np.ndarray] = None
        self.yaxis: Optional[np.ndarray] = None
        self.daxis: Optional[np.ndarray] = None
        self.variable_name: Optional[str] = None
        self.sampler_name: Optional[str] = None

    def read(self, directory: str, post_processor_id: int = 0, variable_index: int = 1):
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
        self.record_log(f"Starting to load data from directory: {directory}", level="INFO")

        # 1. Discover all unique samplers
        all_files = os.listdir(directory)
        sampler_files_0000 = [f for f in all_files if f.endswith('_0000.csv')]

        if not sampler_files_0000:
            msg = f"No samplers found. Could not find files ending in '_0000.csv' in '{directory}'."
            self.record_log(msg, level="ERROR")
            raise FileNotFoundError(msg)

        sampler_base_names = sorted([f.replace('_0000.csv', '') for f in sampler_files_0000])
        self.record_log(f"Discovered {len(sampler_base_names)} samplers: {sampler_base_names}", level="INFO")

        if not (0 <= post_processor_id < len(sampler_base_names)):
            msg = f"post_processor_id {post_processor_id} is out of range. Select an ID between 0 and {len(sampler_base_names) - 1}."
            self.record_log(msg, level="ERROR")
            raise IndexError(msg)

        self.sampler_name = sampler_base_names[post_processor_id]
        self.record_log(f"Selected sampler ID {post_processor_id}: '{self.sampler_name}'", level="INFO")

        # 2. Discover the time series file
        numbered_files = {f for f in all_files if re.match(r'.*?_\d+\.csv$', f)}
        all_csv_files = {f for f in all_files if f.endswith('.csv')}
        time_files = list(all_csv_files - numbered_files)

        if len(time_files) != 1:
            msg = f"Expected 1 time series CSV file, but found {len(time_files)}: {time_files}"
            self.record_log(msg, level="ERROR")
            raise FileNotFoundError(msg)

        time_csv_path = os.path.join(directory, time_files[0])
        self.record_log(f"Discovered time series file: '{time_files[0]}'", level="INFO")

        # 3. Load time vector and the selected vector file series
        try:
            time_df = pd.read_csv(time_csv_path)
            self.taxis = time_df.iloc[:, 0].to_numpy()
        except Exception as e:
            msg = f"Could not load time vector from '{time_csv_path}'. Error: {e}"
            self.record_log(msg, level="ERROR")
            raise ValueError(msg)

        vector_files_pattern = os.path.join(directory, f"{self.sampler_name}_*.csv")
        vector_files = sorted(glob.glob(vector_files_pattern))

        if len(vector_files) != len(self.taxis):
            msg = f"Mismatch: {len(vector_files)} vector files found, but {len(self.taxis)} time steps. Truncating to shorter length."
            self.record_log(msg, level="WARNING")
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
                        self.daxis = np.sqrt(
                            (self.xaxis - self.xaxis[0])**2 + (self.yaxis - self.yaxis[0])**2
                        )
                        self.variable_name = df.columns[variable_index]
                        spatial_axes_established = True
            except Exception as e:
                self.record_log(f"Could not process file {file_path}. Error: {e}", level="WARNING")
                num_spatial_points = len(self.xaxis) if self.xaxis is not None else 0
                slice_data = np.full(num_spatial_points, np.nan)
            data_list.append(slice_data)

        # 5. Finalize the data array
        if not spatial_axes_established:
            self.record_log("All CSV files for this sampler were empty. Data object is empty.", level="WARNING")
            self.data = np.empty((0, len(self.taxis)))
        else:
            full_len = len(self.xaxis)
            for i, arr in enumerate(data_list):
                if len(arr) != full_len:
                    data_list[i] = np.full(full_len, np.nan)

            self.data = np.stack(data_list, axis=1)
            self.record_log(f"Successfully loaded and assembled data. Final shape: {self.data.shape}", level="INFO")

    def write(self, filename: str, *args):
        """
        Write the loaded data to the standard fibeRIS .npz file format.

        Args:
            filename (str): The path to the output .npz file.
            *args: Additional arguments (not used).
        """
        if self.data is None or self.taxis is None or self.xaxis is None or self.yaxis is None:
            raise ValueError("Data is not loaded. Please call read() before writing.")

        if not filename.endswith('.npz'):
            filename += '.npz'

        np.savez(
            filename,
            data=self.data,
            taxis=self.taxis,
            daxis=self.daxis,
            name=self.sampler_name,
            start_time = datetime.datetime.now()
        )
        self.record_log(f"Data successfully written to {filename}.", level="INFO")

    def get_max_indices(self, directory: str) -> tuple[int, int]:
        """
        Get the maximum post_processor_id and variable_index from the data files.

        Args:
            directory (str): The directory containing the MOOSE output CSV files.

        Returns:
            tuple[int, int]: A tuple containing the max_processor_id and max_variable_index.
        """
        self.record_log(f"Getting max indices from directory: {directory}", level="INFO")

        # 1. Discover all unique samplers to find max_processor_id
        all_files = os.listdir(directory)
        sampler_files_0000 = [f for f in all_files if f.endswith('_0000.csv')]

        if not sampler_files_0000:
            msg = f"No samplers found. Could not find files ending in '_0000.csv' in '{directory}'."
            self.record_log(msg, level="ERROR")
            raise FileNotFoundError(msg)

        max_processor_id = len(sampler_files_0000) - 1

        # 2. Read the first sampler's file to find max_variable_index
        try:
            first_sampler_file = sampler_files_0000[0]
            csv_path = os.path.join(directory, first_sampler_file)
            df = pd.read_csv(csv_path)

            if df.shape[1] < 5: # id, var, x, y, z
                 msg = f"File '{first_sampler_file}' has fewer than 5 columns, so no variables available."
                 self.record_log(msg, level="ERROR")
                 raise ValueError(msg)
            
            # The columns are typically [id, var1, var2, ..., x, y, z]
            # The last 3 are coordinates. The first is id.
            # So, valid variable indices are from 1 to shape[1] - 4
            max_variable_index = df.shape[1] - 4

            self.record_log(f"Max processor ID: {max_processor_id}", level="INFO")
            self.record_log(f"Max variable index: {max_variable_index}", level="INFO")

            return max_processor_id, max_variable_index

        except Exception as e:
            msg = f"Could not determine max_variable_index from '{first_sampler_file}'. Error: {e}"
            self.record_log(msg, level="ERROR")
            raise ValueError(msg) from e

