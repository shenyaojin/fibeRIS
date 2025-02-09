# Utils for reading Mariner 1D gauge data
# Shenyao Jin
# I have already packed the data into .npz files, so the .read function is simply reading the
# data from those .npz files.
from itertools import starmap

import numpy as np
from fiberis.analyzer.io import core
from abc import ABC
import os
import datetime


class MarinerGauge1D(core.DataIO, ABC):

    def __init__(self):
        """
        Initialize the gauge data reader
        """
        self.taxis = None
        self.data = None
        self.start_time = None

    def read(self, filename=None):
        """
        Read the gauge data from the npz file. If you want to set the data manually,
        you can use the set_data method.
        :param filename: the filename of the npz file
        :return: None
        """
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