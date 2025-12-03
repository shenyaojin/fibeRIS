# Core class for 2D Tensor processor
# Shenyao Jin, 12/03/2025
# fiberis.analyzer.TensorProcessor.coreT2D

import numpy as np
import datetime
from copy import deepcopy
from typing import Optional, Union, Tuple

from fiberis.utils.history_utils import InfoManagementSystem
from fiberis.analyzer.Data2D.core2D import Data2D


class Tensor2D:
    """
    Class for handling 2D tensor fields that vary over time and a spatial dimension.
    This is designed for data like strain or stress tensor fields along a fiber optic cable.
    """

    def __init__(self,
                 data: Optional[np.ndarray] = None,
                 taxis: Optional[np.ndarray] = None,
                 daxis: Optional[np.ndarray] = None,
                 dim: Optional[int] = None,
                 start_time: Optional[datetime.datetime] = None,
                 name: Optional[str] = None):
        """
        Initialize the Tensor2D class.

        Args:
            data (Optional[np.ndarray]): The tensor data. Expected shape is (n_depth, n_time, dim, dim).
            taxis (Optional[np.ndarray]): 1D array for the time axis (in seconds, relative to start_time).
            daxis (Optional[np.ndarray]): 1D array for the depth/spatial axis.
            dim (Optional[int]): The dimension of the tensor (e.g., 2 for a 2x2 tensor).
            start_time (Optional[datetime.datetime]): Absolute start time of the data.
            name (Optional[str]): Name or identifier for the data.
        """
        self.data: Optional[np.ndarray] = data
        self.taxis: Optional[np.ndarray] = taxis
        self.daxis: Optional[np.ndarray] = daxis
        self.dim: Optional[int] = dim
        self.start_time: Optional[datetime.datetime] = start_time
        self.name: Optional[str] = name
        self.history: InfoManagementSystem = InfoManagementSystem()

        if self.name:
            self.history.add_record(f"Initialized Tensor2D object with name: {self.name}", level="INFO")
        else:
            self.history.add_record("Initialized empty Tensor2D object.", level="INFO")

    def set_data(self, data: np.ndarray) -> 'Tensor2D':
        """
        Set the tensor data. Performs validation.
        The data should have a shape of (n_depth, n_time, dim, dim).
        """
        if not isinstance(data, np.ndarray) or data.ndim != 4:
            raise TypeError("Data must be a 4D NumPy array.")
        if data.shape[2] != data.shape[3]:
            raise ValueError("Tensor dimensions must be square.")

        self.data = data
        if self.dim is None:
            self.dim = data.shape[2]
        self.history.add_record(f"Set data with shape: {self.data.shape}", level="INFO")
        return self

    def set_taxis(self, taxis: np.ndarray) -> 'Tensor2D':
        """Set the time axis."""
        self.taxis = taxis
        self.history.add_record(f"Set taxis with size: {len(self.taxis)}", level="INFO")
        return self

    def set_daxis(self, daxis: np.ndarray) -> 'Tensor2D':
        """Set the depth axis."""
        self.daxis = daxis
        self.history.add_record(f"Set daxis with size: {len(self.daxis)}", level="INFO")
        return self

    def set_dim(self, dim: int) -> 'Tensor2D':
        """Set the tensor dimension."""
        self.dim = dim
        self.history.add_record(f"Set dim to: {self.dim}", level="INFO")
        return self

    def set_start_time(self, start_time: datetime.datetime) -> 'Tensor2D':
        """Set the start time."""
        self.start_time = start_time
        self.history.add_record(f"Set start_time to: {start_time}", level="INFO")
        return self

    def set_name(self, name: str) -> 'Tensor2D':
        """Set the name."""
        self.name = name
        self.history.add_record(f"Set name to: {name}", level="INFO")
        return self

    def load_npz(self, filename: str) -> 'Tensor2D':
        """Load data from a .npz file."""
        if not filename.endswith('.npz'):
            filename += '.npz'
        with np.load(filename, allow_pickle=True) as loader:
            self.set_data(loader['data'])
            self.set_taxis(loader['taxis'])
            self.set_daxis(loader['daxis'])
            self.set_dim(int(loader['dim']))
            if 'start_time' in loader and loader['start_time'].item() is not None:
                self.set_start_time(loader['start_time'].item())
            if 'name' in loader and loader['name'].item() is not None:
                self.set_name(str(loader['name'].item()))
        self.history.add_record(f"Loaded data from {filename}", level="INFO")
        return self

    def savez(self, filename: str) -> None:
        """Save data to a .npz file."""
        if self.data is None or self.taxis is None or self.daxis is None or self.dim is None:
            raise ValueError("Data, taxis, daxis, and dim must be set before saving.")
        if not filename.endswith('.npz'):
            filename += '.npz'
        np.savez(filename,
                   data=self.data,
                   taxis=self.taxis,
                   daxis=self.daxis,
                   dim=self.dim,
                   start_time=self.start_time,
                   name=self.name)
        self.history.add_record(f"Saved data to {filename}", level="INFO")

    def rotate_tensor(self, degree: float, in_radians: bool = False) -> 'Tensor2D':
        """
        Rotates the tensor data in-place by a given angle.
        """
        if not in_radians:
            angle = np.deg2rad(degree)
        else:
            angle = degree

        if self.data is None or self.dim is None:
            raise ValueError("Data or dimension is not set.")

        cos_theta = np.cos(angle)
        sin_theta = np.sin(angle)

        if self.dim == 2:
            R = np.array([[cos_theta, -sin_theta],
                          [sin_theta,  cos_theta]])
        else:
            raise ValueError(f"Unsupported tensor dimension for rotation: {self.dim}.")

        # Using einsum for efficient batch rotation
        self.data = np.einsum('ik,dtkj,lj->dtil', R, self.data, R.T)
        self.history.add_record(f"Rotated tensor data by {degree:.2f} degrees.", level="INFO")
        return self

    def select_time(self, start: Union[datetime.datetime, float, int], end: Union[datetime.datetime, float, int]) -> 'Tensor2D':
        """Crop data along the time axis."""
        if self.start_time is None or self.taxis is None or self.data is None:
            raise ValueError("start_time, taxis, or data is not set.")

        if isinstance(start, (int, float)):
            start_seconds = start
        else:
            start_seconds = (start - self.start_time).total_seconds()

        if isinstance(end, (int, float)):
            end_seconds = end
        else:
            end_seconds = (end - self.start_time).total_seconds()

        time_mask = (self.taxis >= start_seconds) & (self.taxis <= end_seconds)
        self.data = self.data[:, time_mask, :, :]
        self.taxis = self.taxis[time_mask]
        self.history.add_record(f"Selected time from {start_seconds}s to {end_seconds}s.", level="INFO")
        return self

    def select_depth(self, start_depth: float, end_depth: float) -> 'Tensor2D':
        """Crop data along the depth axis."""
        if self.daxis is None or self.data is None:
            raise ValueError("daxis or data is not set.")

        depth_mask = (self.daxis >= start_depth) & (self.daxis <= end_depth)
        self.data = self.data[depth_mask, :, :, :]
        self.daxis = self.daxis[depth_mask]
        self.history.add_record(f"Selected depth from {start_depth} to {end_depth}.", level="INFO")
        return self

    def get_component(self, component: Union[str, Tuple[int, int]]) -> Data2D:
        """
        Extract a single component of the tensor field as a Data2D object.
        """
        if self.data is None:
            raise ValueError("Data is not set.")

        if isinstance(component, str):
            comp_map = {'xx': (0, 0), 'xy': (0, 1), 'yx': (1, 0), 'yy': (1, 1)}
            if component not in comp_map:
                raise ValueError(f"Invalid component string: {component}")
            i, j = comp_map[component]
        else:
            i, j = component

        component_data = self.data[:, :, i, j]
        data2d_obj = Data2D(data=component_data,
                              taxis=self.taxis,
                              daxis=self.daxis,
                              start_time=self.start_time,
                              name=f"{self.name}_{component}")
        self.history.add_record(f"Extracted component ({i},{j}) as Data2D object.", level="INFO")
        return data2d_obj

    def __str__(self) -> str:
        """Return a summary string of the Tensor2D object."""
        info = [f"--- Tensor2D Object: {self.name or 'Unnamed'} ---"]
        info.append(f"  - Data Shape: {self.data.shape if self.data is not None else 'N/A'}")
        info.append(f"  - Time Axis Size: {len(self.taxis) if self.taxis is not None else 'N/A'}")
        info.append(f"  - Depth Axis Size: {len(self.daxis) if self.daxis is not None else 'N/A'}")
        info.append(f"  - Tensor Dimension: {self.dim or 'N/A'}")
        info.append(f"  - Start Time: {self.start_time or 'N/A'}")
        return "\n".join(info)
