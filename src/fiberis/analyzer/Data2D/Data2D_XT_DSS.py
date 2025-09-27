# DSS 2D class from core2D
# Shenyao Jin, shenyaojin@Mines.edu, 02/2025
from fiberis.analyzer.Data2D import core2D
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Tuple

class DSS2D(core2D.Data2D):

    def simple_plot(self, cmap='viridis', title=None, xlabel='Time', ylabel='Depth'):
        """
        Create a simple waterfall plot of the data using imshow.

        Parameters:
        ----------
        cmap : str, optional
            The colormap to use for the plot. Default is 'viridis'.
        title : str, optional
            The title of the plot. Default is 'Simple Plot'.
        xlabel : str, optional
            The label for the x-axis. Default is 'Time'.
        ylabel : str, optional
            The label for the y-axis. Default is 'Depth'.

        Usage:
        ------
        >>> instance = DSS2D()
        >>> instance.simple_plot(cmap='plasma')
        """
        if title is None and self.name is not None:
            title = self.name
        elif title is None:
            title = 'Simple Waterfall Plot'

        plt.figure(figsize=(8, 6))
        plt.imshow(self.data, cmap=cmap, aspect='auto',
                   extent=[self.taxis[0], self.taxis[-1], self.daxis[-1], self.daxis[0]])
        plt.colorbar(label='Amplitude')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def downsampling(self, factor_t=2, factor_d=2) -> None:
        """
        Downsample the data by a specified factor in time and depth dimensions.

        Parameters:
        ----------
        factor_t : int, optional
            The downsampling factor for the time axis. Default is 2.
        factor_d : int, optional
            The downsampling factor for the depth axis. Default is 2.

        Usage:
        ------
        >>> instance = DSS2D()
        >>> instance.downsampling(factor_t=4, factor_d=4)
        """
        self.data = self.data[::factor_d, ::factor_t]
        self.taxis = self.taxis[::factor_t]
        self.daxis = self.daxis[::factor_d]

        self.history.add_record("Downsampled data by factor_t={} and factor_d={}".format(factor_t, factor_d))

