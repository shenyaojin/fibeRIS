# fibeRIS/src/fiberis/io/reader_gold4pb_3d.py
# Reader for Gold 4 PB well geometry data from CSV.
# Shenyao Jin, 11/21/2025

import numpy as np
import pandas as pd
import os
from typing import Any

from fiberis.io import core
from fiberis.analyzer.Geometry3D.coreG3D import DataG3D


class Gold4PB3D(core.DataIO):
    """
    DataIO class for reading 3D well geometry for Gold 4 PB from a CSV file.
    """

    def __init__(self):
        """
        Initializes the Gold4PB3D reader object.
        """
        super().__init__()
        self.xaxis = None
        self.yaxis = None
        self.zaxis = None
        # 'data' will store the Measured Depth (MD)
        self.data = None

    def read(self, filename: str, **kwargs: Any) -> None:
        """
        Reads the geometry data from the specified CSV file.

        The CSV is expected to have the following columns:
        'MD', 'TVDrkb', 'x_gold', 'y_gold'

        - 'x_gold' is mapped to the x-axis.
        - 'y_gold' is mapped to the y-axis.
        - 'TVDrkb' is mapped to the z-axis (True Vertical Depth).
        - 'MD' is mapped to the data attribute (Measured Depth).

        :param filename: str, path to the CSV file.
        :param kwargs: Any, additional keyword arguments (not used).
        """
        self.filename = filename
        try:
            df = pd.read_csv(filename)
        except FileNotFoundError:
            self.record_log(f"Error: File not found at {filename}", level="ERROR")
            raise
        except Exception as e:
            self.record_log(f"An error occurred while reading the CSV file: {e}", level="ERROR")
            raise

        # Assign columns to the corresponding axes
        self.xaxis = df['x_gold'].to_numpy()
        self.yaxis = df['y_gold'].to_numpy()
        self.zaxis = df['TVDrkb'].to_numpy()
        self.data = df['MD'].to_numpy()
        
        self.record_log(f"Successfully read geometry data from {filename}.")

    def write(self, filename: str, *args: Any) -> None:
        """
        Write the loaded geometry data to a standard fibeRIS .npz file.
        The keys 'ew', 'ns', and 'tvd' are used for consistency with other readers.

        :param filename: The filename (path) to save the NPZ file.
        :param args: Any, additional arguments (not used).
        """
        if self.data is None or self.xaxis is None or self.yaxis is None or self.zaxis is None:
            raise ValueError("Data is not loaded. Call read() first.")

        if not filename.lower().endswith('.npz'):
            filename += '.npz'

        # Use standard key names for interoperability
        np.savez(
            filename,
            data=self.data,
            ew=self.xaxis,
            ns=self.yaxis,
            tvd=self.zaxis
        )
        self.record_log(f"Geometry data successfully written to {filename}.")

    def to_analyzer(self) -> DataG3D:
        """
        Directly creates and populates a DataG3D analyzer object from the loaded data.

        Returns:
            DataG3D: A populated analyzer object ready for use.
        """
        if self.data is None or self.xaxis is None or self.yaxis is None or self.zaxis is None:
            raise ValueError("Data is not loaded. Please call the read() method before creating an analyzer.")

        analyzer = DataG3D()
        analyzer.data = self.data
        analyzer.xaxis = self.xaxis
        analyzer.yaxis = self.yaxis
        analyzer.zaxis = self.zaxis
        
        if self.filename:
            analyzer.name = os.path.basename(self.filename)

        analyzer.history.add_record(f"Data populated from {self.__class__.__name__}.")

        return analyzer
