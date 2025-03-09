# Core G3D for geometry 3D data visualization.
# The G3D data would have xaxis, yaxis,zaxis, and data (in measured depth).
# Shenyao Jin, 03/09/2025

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy, copy
from fiberis.analyzer.utils import history_utils

class DataG3D():

     def __init__(self):
         self.data = None
         self.yaxis = None
         self.xaxis = None
         self.zaxis = None
         self.name = None
         self.history = history_utils.InfoManagementSystem()

     def load_npz(self, filename):

         if filename[-4:] != '.npz':
             filename += '.npz'

         data_stucture = np.load(filename)

         self.data = data_stucture['data']
         self.xaxis = data_stucture['xaxis']
         self.yaxis = data_stucture['yaxis']
         self.zaxis = data_stucture['zaxis']

         # get the file name, without the path and extension
         self.name = filename.split('/')[-1][:-4]
         self.history.add_record("Load data from file %s" % filename)

     # Add plot functions here in the future. Not urgent.