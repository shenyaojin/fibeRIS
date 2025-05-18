# Core G3D for geometry 3D data visualization.
# The G3D data would have xaxis, yaxis,zaxis, and data (in measured depth).
# Shenyao Jin, 03/09/2025

import numpy as np
from fiberis.utils import history_utils

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

     def calculate_md(self):
         """
         Computes and returns the measured depth (MD) for each point,
         defined as the cumulative distance from the first point
         to the current point along the well path.

         :return: 1D numpy array of measured depths
         """
         import numpy as np

         # Differences between consecutive points
         dx = np.diff(self.xaxis)
         dy = np.diff(self.yaxis)
         dz = np.diff(self.zaxis)

         # Euclidean distances between consecutive points
         step_distances = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

         # Cumulative sum of distances; prepend 0 for the first point
         md = np.insert(np.cumsum(step_distances), 0, 0.0)
         return md

     # Add plot functions here in the future. Not urgent.