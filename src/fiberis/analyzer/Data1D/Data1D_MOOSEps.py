# src/fiberis/analyzer/Data1D/Data1D_MOOSEps.py
# Specialized Data1D class for reading MOOSE point sampler output files.
# Shenyao Jin, shenyaojin@mines.edu, 06/23/2025

import os
import re
import pandas as pd
import datetime
from fiberis.analyzer.Data1D.core1D import Data1D
from typing import Optional


class Data1D_MOOSEps(Data1D):
    """
    A specialized Data1D class for handling MOOSE Point Sampler post-processor outputs.

    This class inherits all the processing methods from Data1D and provides a
    convenient `read_csv` method to automatically load the point sampler data
    from a MOOSE simulation output directory.
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize the Data1D_MOOSEps class.

        Args:
            name (Optional[str]): An optional name for the data object.
                                  This will be overwritten by the variable name upon loading.
        """
        # Initialize the parent Data1D class
        super().__init__(name=name)
        self.history.add_record("Initialized Data1D_MOOSEps object.", level="INFO")

    def read_csv(self, folder: str, variable_index: int = 1):
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
        self.history.add_record(f"Attempting to find and read MOOSE point sampler CSV in '{folder}'", level="INFO")

        # 1. Automatically discover the point sampler CSV file
        try:
            # Find all CSV files that are NOT part of a numbered sequence
            all_files = os.listdir(folder)
            numbered_files = {f for f in all_files if re.match(r'.*?_\d+\.csv$', f)}
            all_csv_files = {f for f in all_files if f.endswith('.csv')}
            sampler_files = list(all_csv_files - numbered_files)

            if len(sampler_files) != 1:
                msg = f"Expected to find 1 point sampler CSV, but found {len(sampler_files)}: {sampler_files}"
                self.history.add_record(msg, level="ERROR")
                raise FileNotFoundError(msg)

            csv_path = os.path.join(folder, sampler_files[0])
            self.history.add_record(f"Identified point sampler file: '{sampler_files[0]}'", level="INFO")

        except Exception as e:
            self.history.add_record(f"Error during file discovery: {e}", level="ERROR")
            raise

        # 2. Read the data using pandas
        try:
            df = pd.read_csv(csv_path)

            if df.shape[1] <= variable_index:
                msg = f"variable_index {variable_index} is out of bounds for file with {df.shape[1]} columns."
                self.history.add_record(msg, level="ERROR")
                raise IndexError(msg)

            # The first column is always time
            self.taxis = df.iloc[:, 0].to_numpy()
            # The specified column is the data
            self.data = df.iloc[:, variable_index].to_numpy()

            # Set the name of the Data1D object to the variable's column header
            self.name = df.columns[variable_index]

            # Set a default start_time, as it's not present in the CSV
            self.start_time = datetime.datetime.now()

            self.history.add_record(f"Successfully loaded variable '{self.name}' from {csv_path}", level="INFO")
            self.history.add_record(f"Set start_time to current time: {self.start_time.isoformat()}", level="INFO")

        except Exception as e:
            msg = f"Failed to read or process CSV file at '{csv_path}'. Error: {e}"
            self.history.add_record(msg, level="ERROR")
            raise ValueError(msg) from e