# Core G3D for geometry 3D data visualization.
# The G3D data would have xaxis, yaxis,zaxis, and data (in measured depth).
# Shenyao Jin, 03/09/2025

import numpy as np
from fiberis.utils import history_utils
import matplotlib.pyplot as plt
from typing import Optional, Any
from matplotlib.axes import Axes
# This import is necessary for the '3d' projection
from mpl_toolkits.mplot3d import Axes3D


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

     def savez(self, filename):
         """
         Save the current geometry data to an .npz file.

         :param filename: The path to the .npz file where data will be saved.
         """
         if self.data is None or self.xaxis is None or self.yaxis is None or self.zaxis is None:
             self.history.add_record("Error: Cannot save, essential data attributes are not set.", level="ERROR")
             raise ValueError("Data and axes must be set before saving.")

         if not filename.endswith('.npz'):
             filename += '.npz'

         np.savez(filename,
                  data=self.data,
                  xaxis=self.xaxis,
                  yaxis=self.yaxis,
                  zaxis=self.zaxis)
         self.history.add_record("Save data to file %s" % filename)

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

     def get_info_str(self) -> str:
        """
        Get a summary string of the DataG3D object's attributes.
        For array attributes, it shows up to the first 10 elements.
        """
        info_lines = [f"--- DataG3D Object Summary: {self.name or 'Unnamed'} ---"]

        info_lines.append(f"Name: {self.name if self.name else 'Not set'}")

        if self.data is not None:
            info_lines.append(f"Data (MD): Length={self.data.shape[0]}")
            if self.data.size > 0:
                if self.data.size < 10:
                    info_lines.append(f"  Values (first 10): {self.data[:10]}...")
                else:
                    info_lines.append(f"  Values (first 10): {self.data[:10]}...")
        else:
            info_lines.append("Data (MD): Not set")

        # X Axis
        if self.xaxis is not None:
            info_lines.append(f"X Axis (xaxis): Length={self.xaxis.shape[0]}")
            if self.xaxis.size > 0:
                if self.xaxis.size < 10:
                    info_lines.append(f"  Values (first 10): {self.xaxis[:10]}...")
                else:
                    info_lines.append(f"  Values (first 10): {self.xaxis[:10]}...")
        else:
            info_lines.append("X Axis (xaxis): Not set")

        # Y Axis
        if self.yaxis is not None:
            info_lines.append(f"Y Axis (yaxis): Length={self.yaxis.shape[0]}")
            if self.yaxis.size > 0:
                if self.yaxis.size < 10:
                    info_lines.append(f"  Values (first 10): {self.yaxis[:10]}...")
                else:
                    info_lines.append(f"  Values (first 10): {self.yaxis[:10]}...")
        else:
            info_lines.append("Y Axis (yaxis): Not set")

        # Z Axis
        if self.zaxis is not None:
            info_lines.append(f"Z Axis (zaxis): Length={self.zaxis.shape[0]}")
            if self.zaxis.size > 0:
                if self.zaxis.size < 10:
                    info_lines.append(f"  Values (first 10): {self.zaxis[:10]}...")
                else:
                    info_lines.append(f"  Values (first 10): {self.zaxis[:10]}...")
        else:
            info_lines.append("Z Axis (zaxis): Not set")

        info_lines.append(f"History contains {len(self.history.records)} records.")
        info_lines.append("----------------------------------------------------")
        return "\n".join(info_lines)

     def print_info(self) -> None:
        """Prints a summary of the DataG3D object's attributes."""
        print(self.get_info_str())

     def __str__(self) -> str:
        """Return the summary string of the DataG3D object."""
        return self.get_info_str()

     def plot(self, ax: Optional[Axes] = None, **kwargs: Any):
        """
        Plot the 3D geometry.

        Args:
            ax (Optional[Axes]): Matplotlib axes to plot on. If None, a new figure and 3D axes are created.
                                 If provided, it is assumed to be a 3D axes.
            **kwargs: Additional arguments passed to `ax.plot`.

        Returns:
            The artist object(s) created.
        """
        if self.xaxis is None or self.yaxis is None or self.zaxis is None:
            self.history.add_record("Error: Cannot plot. xaxis, yaxis, or zaxis is not set.", level="ERROR")
            raise ValueError("xaxis, yaxis, or zaxis is not set.")

        new_figure_created = False
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=kwargs.pop('figsize', (8, 6)))
            new_figure_created = True

        # Pop plot-specific parameters from kwargs before passing to ax.plot
        xlabel = kwargs.pop('xlabel', "X coordinate")
        ylabel = kwargs.pop('ylabel', "Y coordinate")
        zlabel = kwargs.pop('zlabel', "Z coordinate")
        title = kwargs.pop('title', None)

        # The plot function will raise an error if ax is not a 3D axes, which is reasonable.
        artist = ax.plot(self.xaxis, self.yaxis, self.zaxis, **kwargs)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)

        if title is not None:
            ax.set_title(title)
        elif new_figure_created and self.name:
            ax.set_title(f"3D Geometry: {self.name}")

        if new_figure_created and not plt.isinteractive():
            plt.show()

        self.history.add_record(
            f"Plot generated for '{self.name if self.name else 'Unnamed DataG3D'}'.",
            level="INFO")

        return artist
