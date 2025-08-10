# `fiberis.analyzer.Data3D`

This package is designed for handling 3D datasets, particularly those originating from numerical simulations like MOOSE, where data is often structured as a series of spatial snapshots over time.

## Core Component

### `core3D.py`: The `Data3D` Class

The `Data3D` class is a specialized tool for reading and organizing data from a series of MOOSE `VectorPostprocessor` CSV files. It reconstructs a 3D dataset representing a variable's evolution over a 2D spatial domain through time.

**Key Attributes:**

-   `data`: A NumPy array holding the variable's values, with a shape of `(n_spatial_points, n_time_points)`.
-   `taxis`: A 1D NumPy array for the time axis (in seconds).
-   `xaxis`: A 1D NumPy array for the x-coordinates of the spatial points.
-   `yaxis`: A 1D NumPy array for the y-coordinates of the spatial points.
-   `variable_name`: The name of the variable loaded from the files.
-   `name`: The base name of the MOOSE sampler that was loaded.
-   `history`: A logging system to track all operations.

**Core Functionality:**

The main feature of the `Data3D` class is the `load_from_csv_series` method. This powerful function automates the complex process of reading a MOOSE simulation output directory:

1.  **Sampler Discovery**: It automatically identifies all unique `VectorPostprocessor` outputs by finding files that end with `_0000.csv`.
2.  **Time Discovery**: It locates the main time-series CSV file, which is not part of a numbered sequence.
3.  **Data Loading**: The user selects which sampler to load (by an index). The class then reads the time vector and proceeds to load and concatenate the data from every corresponding numbered CSV file (e.g., `sampler_name_0001.csv`, `sampler_name_0002.csv`, etc.).
4.  **Data Assembly**: It assembles the data from each file into a final `(space, time)` data matrix and extracts the spatial coordinates (`x`, `y`) and the variable name from the files.

## Workflow

The primary use case for this package is to post-process MOOSE simulation results:

1.  Instantiate the `Data3D` class.
2.  Call the `load_from_csv_series` method, pointing it to the directory containing the MOOSE output CSVs and specifying which post-processor and variable to load.
3.  The resulting `Data3D` object will contain the full time-dependent spatial data, ready for further analysis or visualization.
