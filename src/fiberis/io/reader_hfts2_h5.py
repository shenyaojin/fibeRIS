# Utils for reading HFTS2 data from h5 files
# Shenyao Jin, adopted from Peiyao Li, Ge Jin's codes

import numpy as np
from glob import glob
from fiberis.utils import io_utils
from dateutil import parser
import os

from fiberis.io import core
from typing import Optional, Union, List, Tuple, Any, cast, Callable, Dict
import datetime
from fiberis.analyzer.Data2D.Data2D_XT_DSS import DSS2D
from fiberis.analyzer.Data1D.Data1D_Gauge import Data1DGauge

# Need to self check function in coreT.py

class HFTS2DAS2D(core.DataIO):

    def __init__(self) -> None:
        super().__init__()
        self.temp_dssobject = DSS2D()

    def read(self,
             folderpath: str,
             start_time: Optional[datetime.datetime] = None,
             end_time: Optional[datetime.datetime] = None,
             **kwargs: Any
             ) -> None:
        """
        Load HFTS2 raw DAS data from h5 files in a folderpath. Start and end time are optional, if not provided, all data in the folder will be loaded.

        :param folderpath: Path to the folder containing .h5 files.
        :param start_time: Optional start time to filter files.
        :param end_time: Optional end time to filter files.
        :param kwargs:
        :return: None
        """
        # 1. Find all .h5 files
        h5_files = glob(os.path.join(folderpath, '*.h5'))
        if not h5_files:
            print(f"Warning: No .h5 files found in {folderpath}")
            return

        # 2. Extract timestamps and create a list of (filepath, timestamp)
        file_timestamps = []
        for f_path in h5_files:
            try:
                filename = os.path.basename(f_path)
                # Example: sensor_2019-03-18T201959Z.h5
                timestamp_str = filename.split('_')[1].split('.')[0]
                # The 'Z' at the end means UTC (Zulu time)
                dt_obj = parser.parse(timestamp_str)
                file_timestamps.append((f_path, dt_obj))
            except (IndexError, ValueError) as e:
                print(f"Could not parse timestamp from filename: {f_path}. Skipping. Error: {e}")
                continue

        # 3. Sort files chronologically
        file_timestamps.sort(key=lambda x: x[1])

        # 4. Filter by start_time and end_time if provided
        if start_time and file_timestamps:
            # Ensure timezone awareness matches if one is aware and other is naive
            if start_time.tzinfo is None and file_timestamps[0][1].tzinfo is not None:
                start_time = start_time.replace(tzinfo=file_timestamps[0][1].tzinfo)
            file_timestamps = [ft for ft in file_timestamps if ft[1] >= start_time]
        if end_time and file_timestamps:
            if end_time.tzinfo is None and file_timestamps[0][1].tzinfo is not None:
                end_time = end_time.replace(tzinfo=file_timestamps[0][1].tzinfo)
            file_timestamps = [ft for ft in file_timestamps if ft[1] <= end_time]

        if not file_timestamps:
            print("Warning: No files left to process after time filtering.")
            return

        # 5. Load the external depth table
        depth_table_path = os.path.join(folderpath, 'All_DAS_fibre_depth_table.csvh')
        try:
            calibrated_daxis = io_utils.load_hfts2_depthtable(depth_table_path)
            if calibrated_daxis.size == 0:
                raise FileNotFoundError # Treat empty daxis as if file not found
        except FileNotFoundError:
            print(f"Error: Depth table '{depth_table_path}' not found or is empty.")
            print("Cannot proceed without a valid depth axis.")
            return

        # 6. Read and merge files. The first file will initialize the main DSS object.
        for i, (filepath, file_start_time) in enumerate(file_timestamps):
            # Load data from H5 file
            data, _, taxis_unix, _ = io_utils.read_h5(filepath) # daxis from h5 is ignored

            if data is None or taxis_unix is None or taxis_unix.size == 0:
                print(f"Warning: Could not read complete data from {filepath} or taxis is empty. Skipping.")
                continue

            # Validate daxis length against data shape
            if data.shape[0] != calibrated_daxis.shape[0]:
                print(f"Error: Mismatch between data depth dimension ({data.shape[0]}) and calibrated daxis length ({calibrated_daxis.shape[0]}) for file {filepath}.")
                print("Aborting merge process.")
                return

            # Convert timestamps from microseconds to seconds
            taxis_unix_seconds = taxis_unix / 1e6

            # The absolute start time is the first timestamp in the data.
            # The 'Z' in the filename indicates UTC, so we should assume UTC for the data timestamps.
            chunk_start_time = datetime.datetime.fromtimestamp(taxis_unix_seconds[0], tz=datetime.timezone.utc)

            # Calculate relative time axis starting from zero
            taxis_relative = taxis_unix_seconds - taxis_unix_seconds[0]


            # Create a temporary DSS2D object for the current file
            temp_dss = DSS2D(
                data=data,
                daxis=calibrated_daxis, # Use the calibrated daxis
                taxis=taxis_relative,
                start_time=chunk_start_time,
                name=os.path.basename(filepath)
            )

            # If it's the first file, assign it to temp_dssobject. Otherwise, merge.
            if i == 0:
                self.temp_dssobject = temp_dss
                self.temp_dssobject.set_name("HFTS2 Merged Data")
            else:
                try:
                    self.temp_dssobject.right_merge(temp_dss)
                except ValueError as e:
                    print(f"Error merging file {filepath}: {e}")
                    print("Aborting merge process.")
                    return
            
            print(f"Merged file {i+1}/{len(file_timestamps)}: {os.path.basename(filepath)}")

        print("Finished reading and merging all files.")

    def write(self, filename, *args):
        pass

    def to_analyzer(self, **kwargs) -> DSS2D:
        """
        Convert the reader object to a Data2D_XT_DSS.DSS2D object for analysis.

        :return: Data2D_XT_DSS object.
        """
        return self.temp_dssobject

class HFTS2CURVE(core.DataIO):
    """
    Class for loading the
    """

    def __init__(self) -> None:
        super().__init__()

    def read(self, filepath: str, var_name: str) -> None:
        pass

    def write(self, filename, *args) -> None:
        pass

    def to_analyzer(self, **kwargs) -> Data1DGauge:
        pass