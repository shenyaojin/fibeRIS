# Utils for XOM - Bakken Mariner - DSS-RFS data
# Shenyao Jin, shenyaojin@mines.edu

import numpy as np
from fiberis.io import core


class Mariner2DRFS2D(core.DataIO):

    def read(self, **kwargs):
        """

        :param kwargs: mode = h5py, npz(dataio v2); filename = file name;
        :return: None
        """

        pass

    def write(self, filename, *args):
        """
                :param filename: the filename of the npz file
                :param kwargs: format options. In the future I will add csv format support
                :return: None
                """

        if filename[-4:] != '.npz':
            filename += '.npz'

        np.savez(filename, data=self.data, taxis=self.taxis, start_time=self.start_time, daxis=self.daxis)