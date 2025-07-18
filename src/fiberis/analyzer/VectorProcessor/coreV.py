# Core class for vector processor
# Shenyao Jin, 07/18/2025
# fiberis.analyzer.VectorProcessor.coreV

import numpy as np
import matplotlib.pyplot as plt
import datetime
from copy import deepcopy
from typing import Optional, Union, List, Any, Dict

# Fiberis imports
from fiberis.utils.history_utils import InfoManagementSystem



class CoreVector:
    """
    Class for vector processor in fiberis.analyzer.

    Usage: MOOSE module.

    """


    def __init__(self,
                 data: Optional[np.ndarray] = None,
                 taxis: Optional[np.ndarray] = None,
                 dim: Optional[int] = None,
                 start_time: Optional[datetime.datetime] = None,
                 name: Optional[str] = None):
        """
        Intialize the CoreVector class.

        :param data: the tensor data to be processed.
        :param taxis: the time axis of the tensor data. Must start from 0.
        :param dim: the dimension of the tensor data. The dim=3 then the data will have shape of [3, 3, len(taixs)].
        :param start_time: The start time stamp of the data. Must be a datetime object.
        :param name: name of the vector data. If not provided, it will be set to "Vector Data".
        """

        self.data: Optional[np.ndarray] = data
        self.taxis: Optional[np.ndarray] = taxis
        self.dim: Optional[int] = dim
        self.start_time: Optional[datetime.datetime] = start_time
        self.name: Optional[str] = "Default Vector Data" if name is None else name

        self.history: InfoManagementSystem = InfoManagementSystem()

        if name is None:
            self.history.add_record(f"Init CoreVector with default name: {self.name}")
        else:
            self.history.add_record(f"Init CoreVector with name: {self.name}")


    # I/O methods
    def set_data(self, data: np.ndarray):
        """
        Set the tensor data. Performs validation and defensive copying.
        The data should have a shape of [dim, dim, len(taxis)].

        :param data: The tensor data to be set. Must be a numpy array.
        """
        # --- Type and Shape Validation ---
        if not isinstance(data, np.ndarray):
            msg = "data must be a numpy array."
            self.history.add_record(f"Error: {msg}", level='error')
            raise TypeError(msg)
        if data.ndim != 3:
            msg = f"data must be a 3D array, but got {data.ndim} dimensions."
            self.history.add_record(f"Error: {msg}", level='error')
            raise ValueError(msg)
        if data.shape[0] != data.shape[1]:
            msg = f"data must represent square tensors, but shape is [{data.shape[0]}, {data.shape[1]}]."
            self.history.add_record(f"Error: {msg}", level='error')
            raise ValueError(msg)

        # --- Consistency Validation ---
        # Check against existing dimension
        if self.dim is not None and data.shape[0] != self.dim:
            msg = f"Input data dimension ({data.shape[0]}) does not match existing dimension ({self.dim})."
            self.history.add_record(f"Error: {msg}", level='error')
            raise ValueError(msg)
        # Check against existing time axis
        if self.taxis is not None and data.shape[2] != len(self.taxis):
            msg = f"Input data length ({data.shape[2]}) does not match existing taxis length ({len(self.taxis)})."
            self.history.add_record(f"Error: {msg}", level='error')
            raise ValueError(msg)

        # --- Assignment and Logging ---
        # Use a defensive copy
        self.data = data.copy()
        # If dim isn't set, infer it from the data
        if self.dim is None:
            self.set_dim(data.shape[0])

        self.history.add_record(f"Set data with shape: {self.data.shape}")
        self._validate_consistency()
        return self # Allow method chaining

    def set_taxis(self, taxis: np.ndarray):
        """
        Set the time axis. Performs validation and defensive copying.
        The time axis should be a 1D monotonically increasing numpy array.

        :param taxis: The time axis array.
        """
        # --- Type and Shape Validation ---
        if not isinstance(taxis, np.ndarray):
            msg = "taxis must be a numpy array."
            self.history.add_record(f"Error: {msg}", level='error')
            raise TypeError(msg)
        if taxis.ndim != 1:
            msg = f"taxis must be a 1D array, but got {taxis.ndim} dimensions."
            self.history.add_record(f"Error: {msg}", level='error')
            raise ValueError(msg)

        # --- Content Validation ---
        if np.any(np.diff(taxis) < 0):
            msg = "taxis must be monotonically increasing."
            self.history.add_record(f"Error: {msg}", level='error')
            raise ValueError(msg)

        # --- Consistency Validation ---
        if self.data is not None and len(taxis) != self.data.shape[2]:
            msg = f"Input taxis length ({len(taxis)}) does not match existing data length ({self.data.shape[2]})."
            self.history.add_record(f"Error: {msg}", level='error')
            raise ValueError(msg)

        # --- Assignment and Logging ---
        self.taxis = taxis.copy()
        self.history.add_record(f"Set taxis with size: {len(self.taxis)}")
        self._validate_consistency()
        return self # Allow method chaining

    def set_dim(self, dim: int):
        """
        Set the dimension of the tensor.

        :param dim: The dimension to set. Must be an integer.
        """
        if not isinstance(dim, int):
            msg = "dim must be an integer."
            self.history.add_record(f"Error: {msg}", level='error')
            raise TypeError(msg)

        # --- Consistency Validation ---
        if self.data is not None and self.data.shape[0] != dim:
            msg = f"Input dimension ({dim}) does not match existing data's dimension ({self.data.shape[0]})."
            self.history.add_record(f"Error: {msg}", level='error')
            raise ValueError(msg)

        self.dim = dim
        self.history.add_record(f"Set dim to: {self.dim}")
        self._validate_consistency()
        return self # Allow method chaining

    def set_start_time(self, start_time: datetime.datetime):
        """
        Set the start time stamp for the data.

        :param start_time: The start time. Must be a datetime.datetime object.
        """
        if not isinstance(start_time, datetime.datetime):
            msg = "start_time must be a datetime.datetime object."
            self.history.add_record(f"Error: {msg}", level='error')
            raise TypeError(msg)

        self.start_time = start_time
        self.history.add_record(f"Set start_time to: {start_time}")
        return self # Allow method chaining

    def set_name(self, name: str):
        """
        Set the name for the CoreVector instance.

        :param name: The name of the data. Must be a string.
        """
        if not isinstance(name, str):
            msg = "name must be a string."
            self.history.add_record(f"Error: {msg}", level='error')
            raise TypeError(msg)

        self.name = name
        self.history.add_record(f"Set name to: {name}")
        return self # Allow method chaining