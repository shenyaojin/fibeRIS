# `fiberis.analyzer.TensorProcessor`

This package is intended for handling tensor data that evolves over time, which is a common output from physics-based simulations like those in MOOSE (e.g., stress or strain tensors).

## Core Component

### `coreT.py`: The `CoreTensor` Class

The `CoreTensor` class provides a structure for managing time-dependent tensor data. It is designed to hold a sequence of tensors (e.g., 3x3 matrices) at different points in time.

**Key Attributes:**

-   `data`: A 3D NumPy array with a shape of `[dim, dim, len(taxis)]`, where `dim` is the dimension of the tensor (e.g., 3 for a 3x3 stress tensor).
-   `taxis`: A 1D NumPy array for the time axis (in seconds).
-   `dim`: The dimension of the tensor.
-   `start_time`: A `datetime.datetime` object for the absolute start time.
-   `name`: An identifier for the dataset.
-   `history`: A logging system to track operations.

**Core Functionalities:**

-   **I/O**: Includes a `load_npz` method for loading tensor data from a `.npz` file.
-   **Setters**: Provides a suite of setter methods (`set_data`, `set_taxis`, `set_dim`, etc.) for programmatically building the `CoreTensor` object. These methods include validation to ensure data consistency (e.g., matching dimensions between the data array, the `dim` attribute, and the time axis).

## Current Status

As noted in the source code, this class is currently a foundational implementation. The author has indicated that the primary logic for handling tensor-like data was partially implemented within the `Data1D` classes for immediate needs, and this `TensorProcessor` package is a placeholder for more dedicated and complete functionality in the future.