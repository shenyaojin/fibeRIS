# fibeRIS Analyzer

The `fiberis.analyzer` package contains the core data structures for handling various types of engineering and sensing data.

## Modules

### Data1D
Handles one-dimensional time-series data.
*   **Key Class**: `Data1D`
*   **Features**: Time-based cropping, merging, outlier removal, low-pass filtering.
*   **Use Cases**: Pressure gauge data, pumping curves, single-point sensor readings.

### Data2D
Handles two-dimensional spatiotemporal data.
*   **Key Class**: `Data2D`
*   **Features**: Management of data with both time and depth/spatial axes.
*   **Use Cases**: Distributed Acoustic Sensing (DAS) waterfall plots, Distributed Temperature Sensing (DTS).

### Data3D
Handles three-dimensional data.
*   **Key Class**: `Data3D`
*   **Features**: Management of volumetric data or multi-variable datasets with x, y, and time axes.
*   **Use Cases**: Microseismic event clouds, reservoir simulation outputs.

### Data1DG
Handles one-dimensional geometric data.
*   **Key Class**: `Data1DG`
*   **Features**: Spatial axis management (instead of time), range selection, shifting.
*   **Use Cases**: Well logs, depth-based properties.

### Geometry3D
Handles three-dimensional geometric data.
*   **Key Class**: `DataG3D`
*   **Features**: 3D trajectory management (x, y, z), measured depth (MD) calculation.
*   **Use Cases**: Wellbore trajectories, fracture geometries.

### TensorProcessor
Handles tensor data over time.
*   **Key Class**: `CoreTensor`
*   **Features**: Management of n-dimensional tensors (e.g., 3x3 stress tensors), coordinate rotation.
*   **Use Cases**: Geomechanical stress/strain analysis.
