# This script is created to read 2D .dat files from Mariner production data using fiber optics.
# Those .dat files contain 1D daxis, 1D taxis, and 2D data arrays
# NOTE: This *dat file has bad I/O performance, it is recommended to use another one,
# AKA the *.h5 reader, @fibeRIS/src/fiberis/io/reader_mariner_dssh5.py
# Shenyao Jin, 09/09/2025

import numpy as np
import datetime
from fiberis.analyzer.Data2D import Data2D_XT_DSS
from fiberis.io import core
from fiberis.utils.io_utils import read_h5
import os

class MarinerDSSdat2D(core.DataIO):

    def __init__(self):
        """
        I/O function for Mariner 2D .dat files
        """
        super().__init__()

    def read(self, basename: str) -> None:
        """
        Read sets of .dat files from Mariner production data using DFOS

        :param basename: base name of .dat file: in this case, those .dat files contain 1D daxis, 1D taxis, and 2D data
        arrays. They are named like: "./data/POW-H_DTS_depth.dat", "./data/POW-H_DTS_date.dat",
        "./data/POW-H_DTS_data.dat", then the basename is "./data/POW-H_DTS"
        :return: None
        """

        # read 1D daxis
        daxis = np.loadtxt(basename + '_depth.dat')
        # read 1D taxis
        # The delimiter is set to a newline to ensure that each line is read as a single string,
        # preventing splitting on the space between date and time.
        with open(basename + '_date.dat', 'r') as f:
            # Skip the header line and read the rest of the lines, stripping any trailing whitespace
            taxis_str = [line.strip() for line in f.readlines()[1:]]
        # The date format is corrected to '%m/%d/%Y' to match the input file format (e.g., 08/04/2021).
        taxis = np.array([datetime.datetime.strptime(t, '%m/%d/%Y %H:%M:%S.%f') for t in taxis_str])
        # read 2D data
        data = np.loadtxt(basename + '_data.dat')

        self.daxis = daxis.astype(float)
        self.data = data.astype(float)

        # Process the time axis to be relative time in seconds
        t0 = taxis[0]
        self.taxis = np.array([(t - t0).total_seconds() for t in taxis])
        self.start_time = t0

    def write(self, filename, *args) -> None:
        """
        Write data to *.npz file which can be read by fiberis

        :param filename: output filename to the *.npz file
        :param args: additional arguments
        :return: None
        """
        # Rotate the data if needed
        if self.data.shape[0] != len(self.daxis) or self.data.shape[1] != len(self.taxis):
            self.data = self.data.T

        np.savez(filename, daxis=self.daxis, taxis=self.taxis, data=self.data, start_time=self.start_time)

    def to_analyzer(self, **kwargs) -> Data2D_XT_DSS.DSS2D:
        """
        Write data to DSS2D analyzer object for analysis.

        :param kwargs: additional arguments
        :return: DSS2D analyzer object
        """

        # Rotate the data if needed, for future reading DSS
        if self.data.shape[0] != len(self.daxis) or self.data.shape[1] != len(self.taxis):
            self.data = self.data.T

        analyzer = Data2D_XT_DSS.DSS2D()
        analyzer.daxis = self.daxis
        analyzer.taxis = self.taxis
        analyzer.data = self.data
        analyzer.start_time = self.start_time

        if 'filename' in kwargs:
            analyzer.filename = os.path.basename(kwargs['filename'])
        else:
            analyzer.filename = 'Mariner_DSS2D_data'

        analyzer.history.add_record("Data read from Mariner .dat files and converted to DSS2D format.", level='INFO')

        return analyzer