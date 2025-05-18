# Utils for OptaSense - Bakken Mariner - Low Frequency DAS data
# Shenyao Jin, shenyaojin@mines.edu

import numpy as np

from fiberis.io import core


# need to add the self check function
class MarinerDAS2D(core.DataIO):

    def read(self, filename=None):
        """
        Read the gauge data from the npz file.
        :param filename: the filename of the npz file
        :return: None
        """

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

        if filename[-4:] != '.npz':
            filename += '.npz'

        np.savez(filename, data=self.data, taxis=self.taxis, start_time=self.start_time, daxis=self.daxis)