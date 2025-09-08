# Reader function for Mariner DSSH5 files.
# Single file reader.
# Shenyao Jin, 2025-09-04
import datetime

import numpy as np
from fiberis.io import core
from fiberis.analyzer.Data2D import Data2D_XT_DSS
from fiberis.utils.io_utils import read_h5

class MarinerDSS2D(core.DataIO):

    def __init__(self):
        """
        Reader function for Mariner DSSH5 files.
        """
        super().__init__()

    def read(self, filename: str) -> None:
        """
        Read a h5 file and return a reader object.

        :param filename: filepath of the h5 file
        :return: the reader object.
        """

        raw_data, raw_depth, raw_timestamp, start_time = read_h5(filename)

        # Post-process the data
        # For time stamp, convert to seconds relative to start time
        self.taxis = (raw_timestamp - raw_timestamp[0]) / 1e6  # Convert to seconds
        self.daxis = raw_depth
        self.data = raw_data
        # Convert the start time into datetime object
        self.start_time = datetime.datetime.strptime(start_time.decode('utf-8'), '%m/%d/%Y %H:%M:%S.%f')

    def write(self, filename, *args) -> None:
        """
        Write the reader object to a npz file, which can be read by fiberis.

        :param filename: the filename to save the npz file.
        :param args: additional arguments. I don't use it now.
        :return:
        """
        # Rotate the data if needed, for future reading DSS
        if self.data.shape[0] != len(self.daxis) or self.data.shape[1] != len(self.taxis):
            self.data = self.data.T

        np.savez(filename, data=self.data, taxis=self.taxis, daxis=self.daxis, start_time=self.start_time)

    def to_analyzer(self, **kwargs) -> Data2D_XT_DSS:
        """
        Convert the reader object to a Data2D_XT_DSS object for analysis.

        :return: Data2D_XT_DSS object.
        """

        analyzer = Data2D_XT_DSS.DSS2D()
        analyzer.set_data(self.data)
        analyzer.set_taxis(self.taxis)
        analyzer.set_daxis(self.daxis)
        analyzer.set_start_time(self.start_time)

        if 'filename' in kwargs:
            analyzer.set_filename(kwargs['filename'])
        else:
            analyzer.set_filename('Mariner_DSS')

        return analyzer