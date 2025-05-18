# Utils for XOM COOP, Bakken Mariner 3D data reader
# Shenyao Jin, shenyaojin@mines.edu
# created on 3/9/2025
# Write geometry 3D data.

import numpy as np
from fiberis.io import core


class Mariner3D(core.DataIO):

    def __init__(self):
        self.data = None
        self.xaxis = None
        self.yaxis = None
        self.zaxis = None

    def read(self, filename=None):
        dataframe = np.load(filename, allow_pickle=True)
        self.data = dataframe['data']
        self.xaxis = dataframe['ew']
        self.yaxis = dataframe['ns']
        self.zaxis = dataframe['tvd']
        return

    def write(self, filename, *args):
        np.savez(filename,
                 data = self.data,
                 xaxis= self.xaxis,
                 yaxis= self.yaxis,
                 zaxis= self.zaxis)
        return