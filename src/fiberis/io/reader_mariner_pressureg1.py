# fibeRIS/src/fiberis/io/reader_mariner_pressureg1.py

import numpy as np
import pandas as pd
import datetime
import os
from fiberis.io import core
from fiberis.analyzer.Data1D import Data1DGauge

class MarinerPressureG1(core.DataIO):
    """
    DataIO class for reading 1D pressure gauge data from specific CSV files
    and writing them to the standard fibeRIS .npz format.
    """

    def __init__(self):
        """
        Initialize the pressure gauge data reader.
        """
        super().__init__()

    def read(self, filename: str):
        """
        Read the pressure gauge data from a CSV file.

        The CSV file is expected to have two columns: 'datetime' and 'pressure_g1'.
        - 'datetime': Timestamp strings (e.g., '2021-08-02 19:35:50').
        - 'pressure_g1': The pressure data values.

        This method populates the instance's attributes:
        - self.data: NumPy array of pressure values.
        - self.start_time: The first datetime object from the 'datetime' column.
        - self.taxis: NumPy array of elapsed seconds from the start_time.

        Args:
            filename (str): The path to the input CSV file.
        """
        self.filename = filename
        try:
            # Use pandas to efficiently read and parse the CSV data
            df = pd.read_csv(filename)
        except FileNotFoundError:
            print(f"Error: The file {filename} was not found.")
            raise
        except Exception as e:
            print(f"An error occurred while reading the CSV file: {e}")
            raise

        # Check for required columns
        if 'datetime' not in df.columns or 'pressure_g1' not in df.columns:
            raise ValueError(
                f"CSV file '{filename}' must contain 'datetime' and 'pressure_g1' columns."
            )

        # Convert the datetime string column to pandas datetime objects
        df['datetime'] = pd.to_datetime(df['datetime'])

        # Set the start_time to the first timestamp in the series
        self.start_time = df['datetime'].iloc[0].to_pydatetime()

        # Calculate the time axis (taxis) in seconds relative to the start_time
        self.taxis = (df['datetime'] - df['datetime'].iloc[0]).dt.total_seconds().to_numpy()

        # Set the data attribute from the pressure column
        self.data = df['pressure_g1'].to_numpy()

        self.record_log(f"Successfully read data from {filename}.")

    def write(self, filename: str, *args):
        """
        Write the loaded data to the standard fibeRIS .npz file format.

        The output .npz file will contain three arrays:
        - 'data': The pressure data.
        - 'taxis': The time axis in seconds.
        - 'start_time': The absolute start time as a datetime object.

        This format is compatible with `fiberis.analyzer.Data1D.core1D`.

        Args:
            filename (str): The path to the output .npz file.
            *args: Additional arguments (not used in this implementation).
        """
        if self.data is None or self.taxis is None or self.start_time is None:
            raise ValueError(
                "Data is not loaded. Please call the read() method before writing."
            )

        # Ensure the filename ends with .npz
        if not filename.endswith('.npz'):
            filename += '.npz'

        # Save the data arrays to the specified file
        np.savez(
            filename,
            data=self.data,
            taxis=self.taxis,
            start_time=self.start_time
        )
        self.record_log(f"Data successfully written to {filename}.")

    def to_analyzer(self) -> Data1DGauge:
        """
        Directly creates and populates a Data1DGauge analyzer object from the loaded data.

        Returns:
            Data1DGauge: A populated analyzer object ready for use.
        """
        if self.data is None or self.taxis is None or self.start_time is None:
            raise ValueError("Data is not loaded. Please call the read() method before creating an analyzer.")

        analyzer = Data1DGauge()
        analyzer.data = self.data
        analyzer.taxis = self.taxis
        analyzer.start_time = self.start_time
        
        if self.filename:
            analyzer.name = os.path.basename(self.filename)

        analyzer.history.add_record(f"Data populated from {self.__class__.__name__}.")

        return analyzer
