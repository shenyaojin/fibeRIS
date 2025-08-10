# Utils for OptaSense - Bakken Mariner - Low Frequency DAS data
# Shenyao Jin, shenyaojin@mines.edu

import numpy as np
import os
from fiberis.io import core
from fiberis.analyzer.Data2D import DSS2D


class MarinerDAS2D(core.DataIO):

    def __init__(self):
        super().__init__()

    def read(self, filename=None):
        """
        Read the DAS data from the npz file.
        :param filename: the filename of the npz file
        :return: None
        """
        self.filename = filename
        data_structure = np.load(filename, allow_pickle=True)
        self.taxis = data_structure['taxis']
        self.daxis = data_structure['daxis']

        self.data = data_structure['data']
        self.start_time = data_structure['start_time'].item()

    def write(self, filename, *args):
        """
        :param filename: the filename of the npz file
        :param kwargs: format options. In the future I will add csv format support
        :return: None
        """

        if not filename.endswith('.npz'):
            filename += '.npz'

        np.savez(filename, data=self.data, taxis=self.taxis, start_time=self.start_time, daxis=self.daxis)

    def to_analyzer(self) -> DSS2D:
        """
        Directly creates and populates a DSS2D analyzer object from the loaded data.

        Returns:
            DSS2D: A populated analyzer object ready for use.
        """
        if self.data is None or self.taxis is None or self.daxis is None or self.start_time is None:
            raise ValueError("Data is not loaded. Please call the read() method before creating an analyzer.")

        analyzer = DSS2D()
        analyzer.data = self.data
        analyzer.taxis = self.taxis
        analyzer.daxis = self.daxis
        analyzer.start_time = self.start_time
        
        if self.filename:
            analyzer.name = os.path.basename(self.filename)

        analyzer.history.add_record(f"Data populated from {self.__class__.__name__}.")

        return analyzer
