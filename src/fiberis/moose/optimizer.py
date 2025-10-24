# src/moose/optimizer.py
# Description: This file contains the implementation of optimization algorithms for inverse problems.
# Dependencies: MOOSE. Optimizer will not use MOOSE provided optimization module, but will use MOOSE to run simulations.
# Shenyao Jin, 09/24/2025

from typing import List, Dict, Any, Union, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

# import other necessary modules in fiberis
from fiberis.analyzer.Data1D.core1D import Data1D
from fiberis.moose.input_generator import MooseBlock
from fiberis.moose.model_builder import ModelBuilder

class ParameterOptimizer():
    """
    This class provides a high-level tool to run multiple MOOSE instance to analyze specific parameter.
    """

    def __init__(self, project_name: str, sub_model: ModelBuilder):
        """Initialize the optimizer. """
        self.sub_model = sub_model # The input dataframe/file.
