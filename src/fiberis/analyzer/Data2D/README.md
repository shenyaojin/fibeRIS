# `fiberis.analyzer.Data2D`

This package is dedicated to handling two-dimensional datasets, which are frequently encountered in fiber optic sensing. These datasets typically have a spatial axis (like depth or distance along a wellbore) and a time axis, forming a 2D matrix of values.

## Core Component

### `core2D.py`: The `Data2D` Class

The `Data2D` class is the cornerstone of 2D data analysis within `fibeRIS`. It's designed to manage data that can be represented as an image or a grid, such as waterfall plots from Distributed Acoustic Sensing (DAS).

**Key Attributes:**

-   `data`: A 2D NumPy array where dimensions typically correspond to `(depth, time)`.
-   `daxis`: A 1D NumPy array for the spatial or depth axis.
-   `taxis`: A 1D NumPy array for the time axis (in seconds, relative to `start_time`).
-   `start_time`: A `datetime.datetime` object for the absolute start time.
-   `name`: An identifier for the dataset.
-   `history`: A logging system to track all processing steps.

**Core Functionalities:**

-   **I/O**: Loading from and saving to `.npz` files (`load_npz`).
-   **Data Manipulation**:
    -   Cropping along the time axis (`select_time`).
    -   Cropping along the depth axis (`select_depth`).
    -   Time-shifting the entire dataset (`shift`).
    -   Merging datasets chronologically (`right_merge`).
-   **Data Access**: Retrieving a 1D time-series for a specific depth (`get_value_by_depth`).
-   **Signal Processing**: Applying filters along the time axis, such as a low-pass filter (`apply_lowpass_filter`).
-   **Visualization**: A powerful `plot` method that can generate waterfall plots using either `imshow` (for regularly spaced data) or `pcolormesh` (for irregularly spaced data). It supports both relative and absolute time axes.

## Specialized Data Classes

### `Data2D_XT_DSS.py`: The `DSS2D` Class

-   **Inherits from**: `Data2D`
-   **Purpose**: Specifically designed for handling 2D Distributed Strain Sensing (DSS) data, which is often visualized as an XT plot (distance vs. time).
-   **Additional Functionality**: Provides a `simple_plot` method, which is a convenient wrapper around the base class's `plot` method for quickly generating a standard waterfall plot with default settings suitable for DSS data.

## Workflow

1.  Use a reader from `fiberis.io` to load 2D fiber optic data into a `DSS2D` or `Data2D` object.
2.  Use methods like `select_time` and `select_depth` to focus on a region of interest.
3.  Apply signal processing, such as `apply_lowpass_filter`, to enhance the data.
4.  Visualize the data using the `plot` or `simple_plot` methods to create waterfall displays.
5.  Save the processed 2D data object to a standard `.npz` file for future use.
