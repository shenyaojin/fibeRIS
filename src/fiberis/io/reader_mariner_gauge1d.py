# Utils for reading Mariner 1D gauge data
# Shenyao Jin
# I have already packed the data into .npz files, so the .read function is simply reading the
# data from those .npz files.

import numpy as np
from fiberis.io import core
from fiberis.analyzer.Data1D import Data1DGauge
import os


class MarinerGauge1D(core.DataIO):

    def __init__(self):
        """
        Initialize the gauge data reader
        """
        super().__init__()

    def read(self, filename=None):
        """
        Read the gauge data from the npz file. If you want to set the data manually,
        you can use the set_data method.
        :param filename: the filename of the npz file
        :return: None
        """
        self.filename = filename
        data_structure = np.load(filename, allow_pickle=True)
        self.data = data_structure['value']

        taxis_tmp = data_structure['datetime']
        # calculate the time axis for taxis_tmp is in datetime.datetime format
        self.taxis = np.zeros_like(taxis_tmp, dtype=float)
        self.start_time = taxis_tmp[0]
        for i in range(len(taxis_tmp)):
            self.taxis[i] = (taxis_tmp[i] - self.start_time).total_seconds()


    def write(self, filename, **kwargs):
        """

        :param filename: the filename of the npz file
        :param kwargs: format options. In the future I will add csv format support
        :return: None
        """

        if filename[-4:] != '.npz':
            filename += '.npz'

        # save the data. In this case the data will only be read from fiberis.analyzer.data1d

        np.savez(filename, data=self.data, taxis=self.taxis, start_time=self.start_time)

    def to_analyzer(self) -> Data1DGauge:
        """
        Directly creates and populates a Data1DGauge analyzer object from the loaded data.

        Returns:
            Data1DGauge: A populated analyzer object ready for use.
        
        Raises:
            ValueError: If data has not been loaded by calling read() first.
        """
        if self.data is None or self.taxis is None or self.start_time is None:
            raise ValueError("Data is not loaded. Please call the read() method before creating an analyzer.")

        # 1. Instantiate the corresponding analyzer
        gauge_analyzer = Data1DGauge()

        # 2. Transfer data and metadata
        gauge_analyzer.data = self.data
        gauge_analyzer.taxis = self.taxis
        gauge_analyzer.start_time = self.start_time
        
        if self.filename:
             gauge_analyzer.name = os.path.basename(self.filename)

        gauge_analyzer.history.add_record(f"Data populated from {self.__class__.__name__}.")

        return gauge_analyzer
