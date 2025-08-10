# src/fiberis/analyzer/Data1D/Data1D_MOOSEps.py
# Specialized Data1D class for reading MOOSE point sampler output files.
# MOOSE (Point Sampler) handler.
# Shenyao Jin, shenyaojin@mines.edu, 06/23/2025

import os
import re
import pandas as pd
import datetime
from fiberis.analyzer.Data1D.core1D import Data1D
from typing import Optional


class Data1D_MOOSEps(Data1D):
    """
    A specialized Data1D class for handling MOOSE Point Sampler post-processor outputs.

    This class inherits all the processing methods from Data1D. It is intended to be
    used after converting raw MOOSE CSV data to the standard .npz format using the
    `fiberis.io.reader_moose_ps.MOOSEPointSamplerReader`.
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize the Data1D_MOOSEps class.

        Args:
            name (Optional[str]): An optional name for the data object.
        """
        # Initialize the parent Data1D class
        super().__init__(name=name)
        self.history.add_record("Initialized Data1D_MOOSEps object.", level="INFO")