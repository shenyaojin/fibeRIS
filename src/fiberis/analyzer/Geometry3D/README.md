# `fiberis.analyzer.Geometry3D`

This package is focused on handling 3D geometrical data, such as the trajectory of a wellbore. It provides tools to load, process, and analyze spatial path data.

## Core Component

### `coreG3D.py`: The `DataG3D` Class

The `DataG3D` class is the base for representing 3D geometric paths. It stores the spatial coordinates and associated data values.

**Key Attributes:**

-   `data`: A NumPy array that often represents the measured depth (MD) or another value associated with each point along the geometry.
-   `xaxis`, `yaxis`, `zaxis`: 1D NumPy arrays for the X, Y, and Z coordinates of the points defining the path.
-   `name`: An identifier for the dataset.
-   `history`: A logging system to track operations.

**Core Functionalities:**

-   **I/O**: `load_npz` for loading geometry data from a standard `.npz` file.
-   **Calculation**: `calculate_md` computes the measured depth along the well path by calculating the cumulative Euclidean distance between consecutive points.

## Specialized Data Classes

### `DataG3D_md.py`: The `G3DMeasuredDepth` Class

-   **Inherits from**: `DataG3D`
-   **Purpose**: A specialized class for handling 3D geometry where the primary data type is measured depth.
-   **Additional Functionality**: Includes placeholder methods for future features, such as `plot_loc` for visualization and `get_spatial_coor` for retrieving coordinates at a specific measured depth.

## Workflow

1.  Use a reader from `fiberis.io` (like `reader_mariner_3d.py`) to load well trajectory data from a source file into a `.npz` format.
2.  Load this `.npz` file into a `DataG3D` or `G3DMeasuredDepth` object.
3.  Use the `calculate_md` method to compute the measured depth if it's not already the primary data.
4.  (Future) Use plotting and data retrieval methods for analysis.
