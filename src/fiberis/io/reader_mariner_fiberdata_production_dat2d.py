# This script is created to read 2D .dat files from Mariner production data using fiber optics.
# Those .dat files contain 1D daxis, 1D taxis, and 2D data arrays
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
        taxis_str = np.loadtxt(basename + '_date.dat', dtype=str)
        taxis = np.array([datetime.datetime.strptime(t, '%d/%m/%Y %H:%M:%S.%f') for t in taxis_str])
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
        Write data to *.npz file (not implemented)

        :param filename:
        :param args:
        :return:
        """
        pass

    def to_analyzer(self, **kwargs) -> Data2D_XT_DSS.DSS2D:
        """
        Write data to DSS2D analyzer object for analysis (not implemented)

        :param kwargs: additional arguments
        :return: DSS2D analyzer object
        """
        pass