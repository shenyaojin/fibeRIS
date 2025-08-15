# src/fiberis/io/reader_moose_ps.py
# Reader for MOOSE Point Sampler post-processor outputs.
# This class encapsulates the logic for finding and reading the correct CSV file
# from a MOOSE simulation directory and converting it to the standard fibeRIS npz format.
# Shenyao Jin, shenyaojin@mines.edu, 08/09/2025

import os
import re
import pandas as pd
import datetime
import numpy as np
from fiberis.io import core
from typing import Optional

class MOOSEPointSamplerReader(core.DataIO):
    """
    A reader for handling MOOSE Point Sampler post-processor outputs.
    """

    def __init__(self):
        """
        Initialize the MOOSEPointSamplerReader class.
        """
        super().__init__()
        self.variable_name: Optional[str] = None

    def read(self, folder: str, variable_index: int = 1):
        """
        Automatically finds and reads a MOOSE Point Sampler CSV file from a directory.

        It identifies the correct file by finding the single CSV that is NOT part
        of a numbered vector sampler series (i.e., does not end in '_xxxx.csv').

        Args:
            folder (str): The path to the directory containing the MOOSE output files.
            variable_index (int): The column index of the variable to load as data.
                                  Index 0 is the time column. Defaults to 1 (the first variable).

        Raises:
            FileNotFoundError: If the correct point sampler CSV file cannot be uniquely identified.
            IndexError: If the specified variable_index is out of bounds for the CSV file.
            ValueError: If the CSV file cannot be read or is improperly formatted.
        """
        self.record_log(f"Attempting to find and read MOOSE point sampler CSV in '{folder}'", level="INFO")

        # 1. Automatically discover the point sampler CSV file
        try:
            # Find all CSV files that are NOT part of a numbered sequence
            all_files = os.listdir(folder)
            numbered_files = {f for f in all_files if re.match(r'.*?_\d+\.csv$', f)}
            all_csv_files = {f for f in all_files if f.endswith('.csv')}
            sampler_files = list(all_csv_files - numbered_files)

            if len(sampler_files) != 1:
                msg = f"Expected to find 1 point sampler CSV, but found {len(sampler_files)}: {sampler_files}"
                self.record_log(msg, level="ERROR")
                raise FileNotFoundError(msg)

            csv_path = os.path.join(folder, sampler_files[0])
            self.record_log(f"Identified point sampler file: '{sampler_files[0]}'", level="INFO")

        except Exception as e:
            self.record_log(f"Error during file discovery: {e}", level="ERROR")
            raise

        # 2. Read the data using pandas
        try:
            df = pd.read_csv(csv_path)

            if df.shape[1] <= variable_index:
                msg = f"variable_index {variable_index} is out of bounds for file with {df.shape[1]} columns."
                self.record_log(msg, level="ERROR")
                raise IndexError(msg)

            # The first column is always time
            self.taxis = df.iloc[:, 0].to_numpy()
            # The specified column is the data
            self.data = df.iloc[:, variable_index].to_numpy()

            # Get the variable name from the column header
            self.variable_name = df.columns[variable_index]

            # Set a default start_time, as it's not present in the CSV
            self.start_time = datetime.datetime.now()

            self.record_log(f"Successfully loaded variable '{self.variable_name}' from {csv_path}", level="INFO")
            self.record_log(f"Set start_time to current time: {self.start_time.isoformat()}", level="INFO")

        except Exception as e:
            msg = f"Failed to read or process CSV file at '{csv_path}'. Error: {e}"
            self.record_log(msg, level="ERROR")
            raise ValueError(msg) from e

    def write(self, filename: str, *args):
        """
        Write the loaded data to the standard fibeRIS .npz file format.

        Args:
            filename (str): The path to the output .npz file.
            *args: Additional arguments (not used).
        """
        if self.data is None or self.taxis is None or self.start_time is None:
            raise ValueError("Data is not loaded. Please call read() before writing.")

        if not filename.endswith('.npz'):
            filename += '.npz'

        np.savez(
            filename,
            data=self.data,
            taxis=self.taxis,
            start_time=self.start_time
        )
        self.record_log(f"Data successfully written to {filename}.", level="INFO")

    def get_max_index(self, folder: str) -> int:
        """
        Get the maximum index of the variable in the MOOSE Point Sampler output.

        Args:
            folder (str): The path to the directory containing the MOOSE output files.

        Returns:
            int: The maximum index of the variable.
        """
        self.record_log(f"Attempting to find and read MOOSE point sampler CSV in '{folder}'", level="INFO")

        # 1. Automatically discover the point sampler CSV file
        try:
            # Find all CSV files that are NOT part of a numbered sequence
            all_files = os.listdir(folder)
            numbered_files = {f for f in all_files if re.match(r'.*?_\d+\.csv$', f)}
            all_csv_files = {f for f in all_files if f.endswith('.csv')}
            sampler_files = list(all_csv_files - numbered_files)

            if len(sampler_files) != 1:
                msg = f"Expected to find 1 point sampler CSV, but found {len(sampler_files)}: {sampler_files}"
                self.record_log(msg, level="ERROR")
                raise FileNotFoundError(msg)

            csv_path = os.path.join(folder, sampler_files[0])
            self.record_log(f"Identified point sampler file: '{sampler_files[0]}'", level="INFO")

        except Exception as e:
            self.record_log(f"Error during file discovery: {e}", level="ERROR")
            raise

        # 2. Read the data using pandas
        try:
            df = pd.read_csv(csv_path)

            if df.shape[1] <= 1:
                msg = f"CSV file has no data columns. Found {df.shape[1]} columns."
                self.record_log(msg, level="ERROR")
                raise ValueError(msg)

            # The first column is always time, so the maximum index is the number of data columns minus one
            max_index = df.shape[1] - 1

            self.record_log(f"Maximum index of variable in '{csv_path}' is {max_index}.", level="INFO")
            return max_index

        except Exception as e:
            msg = f"Failed to read or process CSV file at '{csv_path}'. Error: {e}"
            self.record_log(msg, level="ERROR")
            raise ValueError(msg) from e