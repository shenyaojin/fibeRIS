# Pumping curve data
# Shenyao Jin, shenyaojin@mines.edu, 02/06/2025

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
from fiberis.analyzer.Data1D import core1D


class Data1DPumpingCurve(core1D.Data1D):
    """
    Class for handling and analyzing 1D pumping curve data.
    Inherits from core1D.Data1D.
    """

    def get_start_time(self, **kwargs):
        """
        Get the start time of the pumping curve based on the data.

        Parameters:
        ----------
        kwargs : dict
            Additional conditions for determining the start time, e.g., threshold values.

        Returns:
        --------
        datetime
            The calculated or detected start time.
        """
        threshold = kwargs.get('threshold', 0.1)  # Default threshold if not provided
        min_index = 0  # First point where data exceeds threshold

        if self.start_time is not None:
            return self.start_time + timedelta(seconds=self.taxis[min_index])
        else:
            raise ValueError("start_time is not set in the data.")

    def get_end_time(self, usedatetime=True, **kwargs):
        """
        Get the end time of the pumping curve based on the data.

        Parameters:
        ----------
        usedatetime : bool, optional
            If True, returns a datetime object. If False, returns seconds from the start time.
            Default is 'datetime'.
        kwargs : dict
            Additional conditions for determining the end time, e.g., threshold values.

        Returns:
        --------
        datetime or float
            The calculated or detected end time as a datetime or seconds from the start time.
        """
        threshold = kwargs.get('threshold', 0.1)  # Default threshold if not provided
        end_index = len(self.data) - 1  # Find last point where data exceeds threshold
        end_index = len(self.data) - 1

        if self.start_time is not None:
            if usedatetime:
                return self.start_time + timedelta(seconds=self.taxis[end_index])
            else:
                return self.taxis[end_index]
        else:
            raise ValueError("start_time is not set in the data.")
    
    