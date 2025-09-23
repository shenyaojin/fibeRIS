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

        # 5. Initialize the main DSS object
        self.temp_dssobject = DSS2D(name="HFTS2 Merged Data")

        # 6. Read and merge files
        for i, (filepath, file_start_time) in enumerate(file_timestamps):
            # Load data from H5 file
            data, daxis, taxis_unix, _ = io_utils.read_h5(filepath)

            if data is None or daxis is None or taxis_unix is None:
                print(f"Warning: Could not read complete data from {filepath}. Skipping.")
                continue

            # Convert unix timestamps to seconds relative to this file's start time
            taxis_relative = taxis_unix - file_start_time.timestamp()

            # Create a temporary DSS2D object for the current file
            temp_dss = DSS2D(
                data=data,
                daxis=daxis,
                taxis=taxis_relative,
                start_time=file_start_time,
                name=os.path.basename(filepath)
            )

            # Merge it with the main object
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