# Utils for XOM COOP, Bakken Mariner 3D data reader
# Shenyao Jin, shenyaojin@mines.edu
# created on 3/9/2025
# Write geometry 3D data.

import numpy as np
import os
from fiberis.io import core
from fiberis.analyzer.Geometry3D import DataG3D


class Mariner3D(core.DataIO):

    def __init__(self):
        super().__init__()
        self.xaxis = None
        self.yaxis = None
        self.zaxis = None

    def read(self, filename=None):
        self.filename = filename
        dataframe = np.load(filename, allow_pickle=True)
        self.data = dataframe['data']
        self.xaxis = dataframe['ew']
        self.yaxis = dataframe['ns']
        self.zaxis = dataframe['tvd']
        return

    def write(self, filename, *args):
        if not filename.endswith('.npz'):
            filename += '.npz'
        np.savez(filename,
                 data = self.data,
                 xaxis= self.xaxis,
                 yaxis= self.yaxis,
                 zaxis= self.zaxis)
        return

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
