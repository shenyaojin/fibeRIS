# Gauge data class inherited from Data1D
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from copy import deepcopy
import datetime

from fiberis.analyzer.Data1D import core1D

# define an object to store the data
class Data1DGauge(core1D.Data1D):


    def calculate_pressure_dropdown(self, start_time=None, end_time=None):
        # Use GPT to implement the code later.
        return 0