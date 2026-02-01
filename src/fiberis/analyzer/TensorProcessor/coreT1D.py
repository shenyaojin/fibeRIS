# Core class for 1D static Tensor processor
# Shenyao Jin, 01/31/2026
# fiberis.analyzer.TensorProcessor.coreT1D

import numpy as np
import matplotlib.pyplot as plt
import datetime
from copy import deepcopy
from typing import Optional, Union, List, Any, Dict

# Fiberis imports
from fiberis.utils.history_utils import InfoManagementSystem


class Tensor1D:
    """
    Class for handling 1D tensor fields that are static in time.
    This is designed for representing spatially varying, time-invariant tensor properties,
    such as the permeability or stress tensor across different geological layers.
    """

    def __init__(self,
                 data: Optional[np.ndarray] = None,
                 daxis: Optional[np.ndarray] = None,
                 dim: Optional[int] = None,
                 name: Optional[str] = None):
        """
        Initialize the Tensor1D class.

        Args:
            data (Optional[np.ndarray]): The tensor data. Expected shape is (n_points, dim, dim).
            daxis (Optional[np.ndarray]): 1D array for the spatial axis (e.g., depth, distance).
            dim (Optional[int]): The dimension of the tensor (e.g., 2 for a 2x2 tensor).
            name (Optional[str]): Name or identifier for the data.
        """
        self.data: Optional[np.ndarray] = data
        self.daxis: Optional[np.ndarray] = daxis
        self.dim: Optional[int] = dim
        self.name: Optional[str] = "Default Tensor1D Data" if name is None else name
        self.history: InfoManagementSystem = InfoManagementSystem()

        if name is None:
            self.history.add_record(f"Init Tensor1D with default name: {self.name}")
        else:
            self.history.add_record(f"Init Tensor1D with name: {self.name}")

    # --- Setter Methods ---
    def set_data(self, data: np.ndarray) -> 'Tensor1D':
        """
        Set the tensor data. Performs validation and defensive copying.
        The data should have a shape of (n_points, dim, dim).
        """
        if not isinstance(data, np.ndarray) or data.ndim != 3:
            raise TypeError(f"Data must be a 3D NumPy array, but got {data.ndim} dimensions.")
        if data.shape[1] != data.shape[2]:
            raise ValueError(f"Tensor dimensions must be square, but got shape ({data.shape[1]}, {data.shape[2]}).")
        if self.daxis is not None and data.shape[0] != len(self.daxis):
            raise ValueError(f"Data length ({data.shape[0]}) does not match daxis length ({len(self.daxis)}).")

        self.data = data.copy()
        if self.dim is None:
            self.dim = data.shape[1]
        self.history.add_record(f"Set data with shape: {self.data.shape}", level="INFO")
        return self

    def set_daxis(self, daxis: np.ndarray) -> 'Tensor1D':
        """Set the spatial axis (daxis)."""
        if not isinstance(daxis, np.ndarray) or daxis.ndim != 1:
            raise TypeError("daxis must be a 1D NumPy array.")
        if self.data is not None and len(daxis) != self.data.shape[0]:
            raise ValueError(f"daxis length ({len(daxis)}) does not match data length ({self.data.shape[0]}).")

        self.daxis = daxis.copy()
        self.history.add_record(f"Set daxis with size: {len(self.daxis)}", level="INFO")
        return self

    def set_dim(self, dim: int) -> 'Tensor1D':
        """Set the tensor dimension."""
        if not isinstance(dim, int) or dim <= 0:
            raise TypeError("dim must be a positive integer.")
        if self.data is not None and self.data.shape[1] != dim:
            raise ValueError(f"Input dimension ({dim}) does not match data's tensor dimension ({self.data.shape[1]}).")
        self.dim = dim
        self.history.add_record(f"Set dim to: {self.dim}", level="INFO")
        return self

    def set_name(self, name: str) -> 'Tensor1D':
        """Set the name."""
        if not isinstance(name, str):
            raise TypeError("name must be a string.")
        self.name = name
        self.history.add_record(f"Set name to: {name}", level="INFO")
        return self

    # --- I/O Methods ---
    def load_npz(self, filename: str) -> 'Tensor1D':
        """Load data from a .npz file."""
        if not filename.endswith('.npz'):
            filename += '.npz'
        with np.load(filename, allow_pickle=True) as loader:
            self.set_data(loader['data'])
            self.set_daxis(loader['daxis'])
            self.set_dim(int(loader['dim']))
            if 'name' in loader and loader['name'].item() is not None:
                self.set_name(str(loader['name'].item()))
        self.history.add_record(f"Loaded data from {filename}", level="INFO")
        return self

    def savez(self, filename: str) -> None:
        """Save data to a .npz file."""
        if self.data is None or self.daxis is None or self.dim is None:
            raise ValueError("Data, daxis, and dim must be set before saving.")
        if not filename.endswith('.npz'):
            filename += '.npz'
        np.savez(filename,
                   data=self.data,
                   daxis=self.daxis,
                   dim=self.dim,
                   name=self.name)
        self.history.add_record(f"Saved data to {filename}", level="INFO")

    def get_component(self, component: Union[str, Tuple[int, int]]) -> np.ndarray:
        """
        Extract a single component of the tensor field as a 1D numpy array.

        Args:
            component (Union[str, Tuple[int, int]]): The component to extract.
                Can be a string ('xx', 'xy', etc.) or a tuple of indices (0, 0).

        Returns:
            np.ndarray: A 1D array of the component values along the daxis.
        """
        if self.data is None:
            raise ValueError("Data is not set.")
        if self.dim is None or self.dim < 2:
            raise ValueError("Dimension must be 2 or greater.")

        comp_map = {'xx': (0, 0), 'xy': (0, 1), 'yx': (1, 0), 'yy': (1, 1),
                    'zz': (2, 2), 'xz': (0, 2), 'zx': (2, 0), 'yz': (1, 2), 'zy': (2, 1)}

        if isinstance(component, str):
            if component not in comp_map:
                raise ValueError(f"Invalid component string: '{component}'.")
            i, j = comp_map[component]
        else:
            i, j = component

        if i >= self.dim or j >= self.dim:
            raise IndexError(f"Component index ({i},{j}) is out of bounds for tensor dimension {self.dim}.")

        component_data = self.data[:, i, j]
        self.history.add_record(f"Extracted component ({i},{j}).", level="INFO")
        return component_data

    def plot_component(self, component: Union[str, Tuple[int, int]], **kwargs):
        """
        Plots a single tensor component against the spatial axis.

        Args:
            component (Union[str, Tuple[int, int]]): The component to plot.
            **kwargs: Additional keyword arguments passed to plt.plot().
        """
        if self.daxis is None:
            raise ValueError("daxis is not set, cannot plot.")

        component_data = self.get_component(component)
        component_str = f"{component[0]}{component[1]}" if isinstance(component, tuple) else component

        plt.figure(figsize=kwargs.get("figsize", (10, 6)))
        plt.plot(self.daxis, component_data, label=f"Component {component_str}", **kwargs)
        plt.xlabel("Spatial Axis")
        plt.ylabel(f"Tensor Component Value")
        plt.title(f"{self.name or 'Tensor1D'}: Component {component_str}")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.show()

    def __str__(self) -> str:
        """Return a summary string of the Tensor1D object."""
        info = [f"--- Tensor1D Object: {self.name or 'Unnamed'} ---"]
        info.append(f"  - Data Shape: {self.data.shape if self.data is not None else 'N/A'}")
        info.append(f"  - Spatial Axis Size: {len(self.daxis) if self.daxis is not None else 'N/A'}")
        info.append(f"  - Tensor Dimension: {self.dim or 'N/A'}")
        return "\n".join(info)
