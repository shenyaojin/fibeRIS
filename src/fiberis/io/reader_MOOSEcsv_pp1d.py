# MOOSE generated CSV reader for 1D data.
# Shenyao Jin, shenyaojin@mines.edu, 05/18/2025
import numpy as np
import pandas as pd
import datetime
import os
from fiberis.io import core
from fiberis.analyzer.Data1D.core1D import Data1D


class MOOSEcsv_pp1d(core.DataIO):
    """
    DataIO class for reading a specific column from a MOOSE CSV file
    and writing it to the standard .npz format.
    """

    def __init__(self):
        """
        Initializes the MOOSEcsv_pp1d object.
        """
        super().__init__()
        self.label = None
        # Set a default start time, which is often arbitrary for simulation data
        self.start_time = datetime.datetime(2024, 1, 1)

    def list_available_keys(self, filename: str) -> list:
        """
        Reads and prints the available column headers (keys) from the specified CSV file.

        :param filename: str, path to the CSV file.
        :return: list, a list of available data keys (column headers excluding 'time').
        """
        if filename is None:
            raise ValueError("Filename must be provided to list available keys.")

        try:
            headers = pd.read_csv(filename, nrows=0).columns.tolist()
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
            raise
        except pd.errors.EmptyDataError:
            print(f"Error: File '{filename}' is empty or contains no data.")
            return []
        except Exception as e:
            print(f"Error reading CSV file '{filename}' to get headers: {e}")
            raise

        print(f"\nAvailable columns in '{filename}':")
        data_keys = [h for h in headers if h != 'time']
        print(f"  - time (Time axis column)")
        for key in data_keys:
            print(f"  - {key} (Data key)")
        print("-" * 30)
        return data_keys

    def read(self, filename: str, key: str):
        """
        Reads the 'time' column and a specified data column ('key') from a MOOSE CSV file.

        :param filename: str, path to the CSV file.
        :param key: str, the header name of the data column to read (e.g., "pp_mon2").
        """
        self.filename = filename
        if key is None:
            raise ValueError("Data 'key' must be provided to specify which column to read.")

        try:
            df = pd.read_csv(filename)
        except FileNotFoundError:
            raise
        except Exception as e:
            raise

        if 'time' not in df.columns:
            raise ValueError("CSV file must contain a 'time' column.")

        if key not in df.columns:
            raise ValueError(f"Key '{key}' not found in CSV columns. Available columns: {df.columns.tolist()}")

        self.taxis = df['time'].to_numpy()
        self.data = df[key].to_numpy()
        self.label = key

    def write(self, filename: str, **kwargs):
        """
        Write the loaded data to a standard NPZ file.

        :param filename: The filename (path) to save the NPZ file.
        :param kwargs: Reserved for future format options.
        """
        if self.data is None or self.label is None:
            raise ValueError("Data and label are not initialized. Call read() first.")

        if not filename.endswith('.npz'):
            filename += '.npz'

        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        np.savez(
            filename,
            data=self.data,
            taxis=self.taxis,
            start_time=self.start_time,
            label=self.label
        )

    def to_analyzer(self) -> Data1D:
        """
        Directly creates and populates a Data1D analyzer object.

        Returns:
            Data1D: A populated analyzer object ready for use.
        """
        if self.data is None or self.taxis is None or self.start_time is None:
            raise ValueError("Data is not loaded. Please call read() first.")

        analyzer = Data1D()
        analyzer.data = self.data
        analyzer.taxis = self.taxis
        analyzer.start_time = self.start_time
        analyzer.name = self.label

        analyzer.history.add_record(f"Data populated from {self.__class__.__name__}.")

        return analyzer
