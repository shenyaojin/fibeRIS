# fibeRIS/src/fiberis/io/reader_bearskin_injection.py
# Reader for Bearskin injection data from CSV.
# Shenyao Jin, 11/25/2025

import numpy as np
import pandas as pd
import os
from typing import Any

from fiberis.io import core
from fiberis.analyzer.Geometry3D.DataG3D_md import G3DMeasuredDepth


class BearskinInjection(core.DataIO):
    """
    DataIO class for reading 3D injection data for Bearskin from a CSV file.
    """

    def __init__(self):
        """
        Initializes the BearskinInjection reader object.
        """
        super().__init__()
        self.xaxis = None
        self.yaxis = None
        self.zaxis = None
        # 'data' will store the Stimulating Point
        self.data = None

    def read(self, filename: str, **kwargs: Any) -> None:
        """
        Reads the geometry data from the specified CSV file.

        The CSV is expected to have the following columns:
        'X', 'Y', 'Z', 'Stimulating Point'

        - 'X' is mapped to the x-axis.
        - 'Y' is mapped to the y-axis.
        - 'Z' is mapped to the z-axis.
        - 'Stimulating Point' is mapped to the data attribute.

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

        # Assign columns to the corresponding axes, reversing Z
        self.xaxis = df['X'].to_numpy()
        self.yaxis = df['Y'].to_numpy()
        self.zaxis = -df['Z'].to_numpy()
        self.data = df['Stimulating Point'].to_numpy()
        
        self.record_log(f"Successfully read injection data from {filename}.")

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

    def to_analyzer(self) -> G3DMeasuredDepth:
        """
        Directly creates and populates a G3DMeasuredDepth analyzer object from the loaded data.

        Returns:
            G3DMeasuredDepth: A populated analyzer object ready for use.
        """
        if self.data is None or self.xaxis is None or self.yaxis is None or self.zaxis is None:
            raise ValueError("Data is not loaded. Please call the read() method before creating an analyzer.")

        analyzer = G3DMeasuredDepth()
        analyzer.data = self.data
        analyzer.xaxis = self.xaxis
        analyzer.yaxis = self.yaxis
        analyzer.zaxis = self.zaxis
        
        if self.filename:
            analyzer.name = os.path.basename(self.filename)

        analyzer.history.add_record(f"Data populated from {self.__class__.__name__}.")

        return analyzer
