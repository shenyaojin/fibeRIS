# fibeRIS/src/fiberis/io/reader_moose_tensor_vpp.py
# Reader for MOOSE VectorPostProcessor CSV file series for tensor data.
# Shenyao Jin, 12/03/2025

import datetime
import os
import re
import glob
import numpy as np
import pandas as pd
from fiberis.io import core
from typing import Optional, List
from fiberis.analyzer.TensorProcessor.coreT2D import Tensor2D

class MOOSETensorVPPReader(core.DataIO):
    """
    A reader for handling a series of MOOSE VectorPostprocessor CSV files that represent tensor data.
    It auto-discovers strain samplers, reads the xx, yy, and xy components,
    and assembles them into a Tensor2D object.
    """

    def __init__(self):
        """
        Initialize the MOOSETensorVPPReader class.
        """
        super().__init__()
        self.sampler_name: Optional[str] = None

    def list_available_samplers(self, directory: str) -> List[str]:
        """
        Scans the directory for available strain tensor samplers.

        Args:
            directory (str): The directory containing the MOOSE output CSV files.

        Returns:
            List[str]: A list of discoverable sampler base names.
        """
        self.record_log(f"Scanning for available samplers in: {directory}", level="INFO")
        all_files = os.listdir(directory)
        
        # Find files that mark the start of a vector postprocessor series and contain 'strain_sampler'
        sampler_files_0000 = [f for f in all_files if f.endswith('_0000.csv') and 'strain_sampler' in f]

        if not sampler_files_0000:
            self.record_log(f"No strain samplers found in '{directory}'.", level="WARNING")
            return []

        sampler_base_names = sorted([f.replace('_0000.csv', '') for f in sampler_files_0000])
        self.record_log(f"Discovered {len(sampler_base_names)} samplers: {sampler_base_names}", level="INFO")
        return sampler_base_names

    def read(self, directory: str, sampler_name: str):
        """
        Loads tensor data from a specified sampler in a MOOSE output directory.

        Args:
            directory (str): The directory containing the MOOSE output CSV files.
            sampler_name (str): The base name of the sampler to load (e.g., '..._fiber_strain_sampler_100.0ft').
        """
        self.record_log(f"Starting to load data for sampler '{sampler_name}' from directory: {directory}", level="INFO")
        self.sampler_name = sampler_name

        # 1. Discover and load the time series file
        all_files = os.listdir(directory)
        numbered_files = {f for f in all_files if re.match(r'.*?_\d+\.csv$', f)}
        all_csv_files = {f for f in all_files if f.endswith('.csv')}
        time_files = list(all_csv_files - numbered_files)

        if len(time_files) != 1:
            msg = f"Expected 1 time series CSV file, but found {len(time_files)}: {time_files}"
            self.record_log(msg, level="ERROR")
            raise FileNotFoundError(msg)

        time_csv_path = os.path.join(directory, time_files[0])
        self.record_log(f"Discovered time series file: '{time_files[0]}'", level="INFO")

        try:
            time_df = pd.read_csv(time_csv_path)
            self.taxis = time_df.iloc[:, 0].to_numpy()
            self.start_time = datetime.datetime.now() # Default start time
        except Exception as e:
            raise ValueError(f"Could not load time vector from '{time_csv_path}'. Error: {e}")

        # 2. Get the list of vector files for the chosen sampler
        vector_files_pattern = os.path.join(directory, f"{self.sampler_name}_*.csv")
        vector_files = sorted(glob.glob(vector_files_pattern))

        if not vector_files:
            raise FileNotFoundError(f"No CSV files found for sampler '{self.sampler_name}' in '{directory}'.")

        if len(vector_files) != len(self.taxis):
            min_len = min(len(vector_files), len(self.taxis))
            self.record_log(f"Mismatch: {len(vector_files)} vector files vs {len(self.taxis)} time steps. Truncating to {min_len}.", level="WARNING")
            vector_files = vector_files[:min_len]
            self.taxis = self.taxis[:min_len]

        # 3. Process the file series to build the tensor data
        tensor_slices = []
        spatial_axes_established = False
        for file_path in vector_files:
            try:
                df = pd.read_csv(file_path)
                if not spatial_axes_established:
                    x_coords = df['x'].to_numpy()
                    y_coords = df['y'].to_numpy()
                    self.daxis = np.sqrt((x_coords - x_coords[0])**2 + (y_coords - y_coords[0])**2)
                    spatial_axes_established = True

                n_points = len(self.daxis)
                slice_tensor = np.zeros((n_points, 2, 2))
                slice_tensor[:, 0, 0] = df['strain_xx'].to_numpy()
                slice_tensor[:, 1, 1] = df['strain_yy'].to_numpy()
                slice_tensor[:, 0, 1] = df['strain_xy'].to_numpy()
                slice_tensor[:, 1, 0] = df['strain_xy'].to_numpy() # Assuming symmetric tensor
                tensor_slices.append(slice_tensor)

            except Exception as e:
                self.record_log(f"Could not process file {file_path}. Error: {e}", level="WARNING")
                continue
        # 3.5 Post-processing checks
        # I found the data output from MOOSE always has one more taxis point than we expect.
        # remove the last time point to align data dimensions.
        self.taxis = self.taxis[:len(tensor_slices)]
        
        # 4. Stack the time slices into the final data array
        # The shape should be (n_depth, n_time, dim, dim)
        self.data = np.stack(tensor_slices, axis=1)
        self.record_log(f"Successfully assembled tensor data. Final shape: {self.data.shape}", level="INFO")


    def write(self, filename: str, *args):
        """
        Not implemented for this reader. Use the analyzer's savez method.
        """
        self.record_log("Write operation is not supported by this reader. Please use the analyzer's `savez` method.", level="WARNING")
        pass

    def to_analyzer(self) -> Tensor2D:
        """
        Convert the loaded data to a Tensor2D object for analysis.

        Returns:
            Tensor2D: The Tensor2D object containing the loaded data.
        """
        if self.data is None or self.taxis is None or self.daxis is None:
            raise ValueError("Data is not loaded. Please call read() before converting to analyzer.")

        analyzer = Tensor2D(
            data=self.data,
            taxis=self.taxis,
            daxis=self.daxis,
            dim=2,
            start_time=self.start_time,
            name=self.sampler_name
        )
        analyzer.history.add_record(f"Data loaded from MOOSE Tensor VPP files.", level="INFO")
        return analyzer
