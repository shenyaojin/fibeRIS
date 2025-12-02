# Core class for Tensor processor
# Shenyao Jin, 07/18/2025
# fiberis.analyzer.TensorProcessor.coreV

import numpy as np
import matplotlib.pyplot as plt
import datetime
from copy import deepcopy
from typing import Optional, Union, List, Any, Dict

# Fiberis imports
from fiberis.utils.history_utils import InfoManagementSystem


class CoreTensor:
    """
    Class for Tensor processor in fiberis.analyzer. It handles tensor data (like stress or strain tensors) over time.

    If you find any issues email me at shenyaojin@mines.edu.

    Usage: MOOSE module.
    """


    def __init__(self,
                 data: Optional[np.ndarray] = None,
                 taxis: Optional[np.ndarray] = None,
                 dim: Optional[int] = None,
                 start_time: Optional[datetime.datetime] = None,
                 name: Optional[str] = None):
        """
        Intialize the CoreTensor class.

        :param data: the tensor data to be processed.
        :param taxis: the time axis of the tensor data. Must start from 0.
        :param dim: the dimension of the tensor data. The dim=3 then the data will have shape of [3, 3, len(taixs)].
        :param start_time: The start time stamp of the data. Must be a datetime object.
        :param name: name of the Tensor data. If not provided, it will be set to "Tensor Data".
        """

        self.data: Optional[np.ndarray] = data
        self.taxis: Optional[np.ndarray] = taxis
        self.dim: Optional[int] = dim
        self.start_time: Optional[datetime.datetime] = start_time
        self.name: Optional[str] = "Default Tensor Data" if name is None else name

        self.history: InfoManagementSystem = InfoManagementSystem()

        if name is None:
            self.history.add_record(f"Init CoreTensor with default name: {self.name}")
        else:
            self.history.add_record(f"Init CoreTensor with name: {self.name}")


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
        Set the name for the CoreTensor instance.

        :param name: The name of the data. Must be a string.
        """
        if not isinstance(name, str):
            msg = "name must be a string."
            self.history.add_record(f"Error: {msg}", level='error')
            raise TypeError(msg)

        self.name = name
        self.history.add_record(f"Set name to: {name}")
        return self # Allow method chaining

    def load_npz(self, filename: str) -> 'CoreTensor':
        """
        Load data from a .npz file.
        Expected keys: 'data', 'taxis', 'dim'. Optional: 'start_time', 'name'.
        :param filename: Path to the .npz file.
        """
        if not filename.endswith('.npz'):
            filename_ext = filename + ".npz"
        else:
            filename_ext = filename

        self.history.add_record(f"Loading data from {filename_ext}")

        try:
            data_structure = np.load(filename_ext, allow_pickle=True)
        except FileNotFoundError:
            self.history.add_record(f"Error: File {filename_ext} not found.", level='error')
            raise

        try:
            # Required attributes
            self.set_data(data_structure['data'])
            self.set_taxis(data_structure['taxis'])
            self.set_dim(int(data_structure['dim']))

            # Optional attributes
            if 'start_time' in data_structure and data_structure['start_time'].item() is not None:
                start_time_raw = data_structure['start_time'].item()
                if isinstance(start_time_raw, datetime.datetime):
                    self.set_start_time(start_time_raw)

            if 'name' in data_structure and data_structure['name'].item() is not None:
                self.set_name(str(data_structure['name'].item()))

        except KeyError as e:
            self.history.add_record(f"Error: Missing required key in .npz file: {e}", level='error')
            raise KeyError(f"Missing required key in .npz file: {e}")

        self.history.add_record(f"Successfully loaded data from {filename_ext}.", level="INFO")
        return self

    def savez(self, filename: str) -> 'CoreTensor':
        """
        Save the current tensor data to an .npz file.
        :param filename: The path to the .npz file where data will be saved.
        """
        if self.data is None or self.taxis is None or self.dim is None:
            self.history.add_record("Error: Cannot save, essential data attributes are not set.", level="ERROR")
            raise ValueError("Data, taxis, and dim must be set before saving.")

        if not filename.endswith('.npz'):
            filename += '.npz'

        np.savez(
            filename,
            data=self.data,
            taxis=self.taxis,
            dim=self.dim,
            start_time=self.start_time,
            name=self.name
        )
        self.history.add_record(f"Data successfully saved to {filename}.", level="INFO")
        return self

    def rotate_tensor(self, degree: float) -> None:
        """
        Rotates the tensor data in-place by a given degree (in radians).
        For 2D tensors, it performs a standard 2D rotation.
        For 3D tensors, it performs a rotation around the Z-axis.

        :param degree: The rotation angle in radians.
        :raises TypeError: If degree is not a float or int.
        :raises ValueError: If data or dimension is not set, or if dimension is not 2 or 3.
        """
        if not isinstance(degree, (float, int)):
            msg = "Rotation degree must be a float or an integer."
            self.history.add_record(f"Error: {msg}", level='error')
            raise TypeError(msg)

        if self.data is None or self.dim is None:
            msg = "Cannot rotate tensor: data or dimension is not set."
            self.history.add_record(f"Error: {msg}", level='error')
            raise ValueError(msg)

        cos_theta = np.cos(degree)
        sin_theta = np.sin(degree)

        if self.dim == 2:
            # 2D rotation matrix
            R = np.array([[cos_theta, -sin_theta],
                          [sin_theta,  cos_theta]])
        elif self.dim == 3:
            # 3D rotation matrix around Z-axis
            R = np.array([[cos_theta, -sin_theta, 0],
                          [sin_theta,  cos_theta, 0],
                          [0,           0,          1]])
        else:
            msg = f"Unsupported tensor dimension for rotation: {self.dim}. Only 2D and 3D are supported."
            self.history.add_record(f"Error: {msg}", level='error')
            raise ValueError(msg)

        # Pre-calculate transpose for efficiency
        R_transpose = R.T

        # Apply rotation to each tensor slice over time
        for i in range(self.data.shape[2]):
            self.data[:, :, i] = R @ self.data[:, :, i] @ R_transpose

        self.history.add_record(f"Rotated tensor data by {np.degrees(degree):.2f} degrees.", level="INFO")
