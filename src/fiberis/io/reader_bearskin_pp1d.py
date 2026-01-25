# fibeRIS/src/fiberis/io/reader_bearskin_pp1d.py
# Shenyao Jin, 11/21/2025
# Reader for Bearskin pressure data from CSV files.

import numpy as np
import pandas as pd
import datetime
import os
from typing import Optional, Any, List

from fiberis.io import core
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge


class BearskinPP1D(core.DataIO):
    """
    DataIO class for reading 1D pressure data from Bearskin CSV files
    and writing it to the standard fibeRIS .npz format.
    """

    def __init__(self):
        """
        Initialize the Bearskin pressure data reader.
        """
        super().__init__()
        self.well_name: Optional[str] = None
        self.stage_number: Optional[int] = None

    def read(self, file_path: str, stage_num: Optional[int] = None, **kwargs: Any) -> None:
        """
        Read pressure data from a Bearskin CSV file.

        The CSV file is expected to have columns like:
        'TEMPERATURE_F', 'PRESSURE_PSI', 'UTCTIMESTAMPT', 'MSTTIMESTAMP', 'WELLNAME', 'STAGENUMBER'

        This method populates the instance's attributes:
        - self.data: NumPy array of pressure values ('PRESSURE_PSI').
        - self.start_time: The first datetime object from the 'MSTTIMESTAMP' column (local time).
        - self.taxis: NumPy array of elapsed seconds from the start_time.

        Args:
            file_path (str): The path to the input CSV file.
            stage_num (Optional[int]): This parameter is ignored. The entire file is read.
            **kwargs (Any): Additional keyword arguments (not used).
        """
        self.filename = file_path
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            self.record_log(f"Error: The file {file_path} was not found.", level="ERROR")
            raise
        except Exception as e:
            self.record_log(f"An error occurred while reading the CSV file: {e}", level="ERROR")
            raise

        if stage_num is not None:
            self.record_log(
                f"Warning: stage_num={stage_num} was provided but is being ignored. "
                "The entire file will be read as a single dataset.",
                level="WARNING"
            )

        unique_stages = df['STAGENUMBER'].unique()
        if len(unique_stages) > 1:
            self.record_log(
                f"Multiple stages found: {sorted(unique_stages)}. "
                "Reading all as a single dataset. Time gaps between stages may cause discontinuities in plots.",
                level="INFO"
            )

        # Convert MSTTIMESTAMP to datetime objects
        df['MSTTIMESTAMP'] = pd.to_datetime(df['MSTTIMESTAMP'], errors='coerce')
        df.dropna(subset=['MSTTIMESTAMP'], inplace=True)

        if df.empty:
            self.record_log(f"Warning: No valid MSTTIMESTAMP data found in the file.", level="WARNING")
            self.data = np.array([])
            self.taxis = np.array([])
            self.start_time = None
            self.well_name = None
            return

        df = df.sort_values(by='MSTTIMESTAMP').reset_index(drop=True)

        self.start_time = df['MSTTIMESTAMP'].iloc[0].to_pydatetime()
        self.well_name = df['WELLNAME'].iloc[0] if not df.empty else None

        # --- Data Cleansing ---
        initial_rows = len(df)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=['PRESSURE_PSI'], inplace=True)

        cleaned_rows = len(df)
        rows_removed = initial_rows - cleaned_rows
        if rows_removed > 0:
            self.record_log(f"Removed {rows_removed} rows with NaN, Inf, or invalid pressure values.", level="INFO")

        # --- Final Data Assignment ---
        if df.empty:
            self.record_log(
                f"Warning: No valid pressure data found in {file_path} after data cleansing. "
                f"start_time is set to the first timestamp found in the file.",
                level="WARNING"
            )
            self.data = np.array([])
            self.taxis = np.array([])
            return

        self.taxis = (df['MSTTIMESTAMP'] - pd.Timestamp(self.start_time)).dt.total_seconds().to_numpy()
        self.data = df['PRESSURE_PSI'].to_numpy()

        self.record_log(f"Successfully read data from {file_path} for well '{self.well_name}'.")

    def get_all_stages(self, file_path: str) -> List[int]:
        """
        Reads the CSV file and returns a sorted list of unique stage numbers.

        Args:
            file_path (str): The path to the input CSV file.

        Returns:
            List[int]: A sorted list of all unique stage numbers present in the file.
        """
        try:
            df = pd.read_csv(file_path, usecols=['STAGENUMBER'])
            stages = sorted(df['STAGENUMBER'].unique().tolist())
            return stages
        except FileNotFoundError:
            self.record_log(f"Error: The file {file_path} was not found.", level="ERROR")
            raise
        except Exception as e:
            self.record_log(f"An error occurred while reading stages from the CSV file: {e}", level="ERROR")
            raise

    def write(self, filename: str, *args: Any) -> None:
        """
        Write the loaded data to the standard fibeRIS .npz file format.

        Args:
            filename (str): The path to the output .npz file.
            *args (Any): Additional arguments (not used).
        """
        if self.data is None or self.taxis is None or self.start_time is None:
            raise ValueError("Data is not loaded. Please call the read() method before writing.")

        if not filename.lower().endswith('.npz'):
            filename += '.npz'

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
            raise ValueError("Data is not loaded. Please call read() before creating an analyzer.")

        analyzer = Data1DGauge()
        analyzer.data = self.data
        analyzer.taxis = self.taxis
        analyzer.start_time = self.start_time

        name_parts = []
        if self.filename:
            name_parts.append(os.path.basename(self.filename).replace('.csv', ''))
        if self.well_name:
            name_parts.append(self.well_name)
        if self.stage_number is not None:
            name_parts.append(f"Stage_{self.stage_number}")
        analyzer.name = '_'.join(name_parts)

        analyzer.history.add_record(f"Data populated from {self.__class__.__name__}.")
        return analyzer
