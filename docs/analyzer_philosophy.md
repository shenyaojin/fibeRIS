This note I'll the philosophy of the `fiberis.analyzer`.

# Introduction

The `fiberis.analyzer` is a submodule of the `fibeRIS`, which is responsible for all the data management and analysis in the `fibeRIS`. It provides a set of tools and utilities to facilitate the analysis of simulation results, including data extraction, visualization, and post-processing.

This document outlines the core design philosophy and provides a guide for developers contributing to this module. Adhering to these principles is crucial for maintaining the consistency, reliability, and usability of the package.

# Core Philosophy

The `analyzer` module is built on a few fundamental concepts. Understanding these is key to developing new functionality correctly.

### 1. Data-Centric Object Model

The central idea is to move away from passing around raw NumPy arrays and instead use rich, data-centric objects. Each object acts as a container not just for the data, but also for its essential metadata and history.

-   **Data**: The raw numerical data, typically a NumPy array (`self.data`).
-   **Axes**: The physical coordinates of the data, such as time, depth, or spatial location (`self.taxis`, `self.daxis`, `self.xaxis`, etc.).
-   **Metadata**: Essential information required to understand the data, most importantly the absolute `start_time` (a `datetime.datetime` object) and a `name` for the dataset.
-   **History**: A complete, auditable log of every operation performed on the object since its creation.

### 2. Dimensionality as the Primary Organizer

The entire `analyzer` package is structured by the dimensionality of the data it handles.

-   `Data1D`: For 1D time-series data (e.g., a single gauge reading over time).
-   `Data2D`: For 2D matrix data (e.g., fiber optic data with axes of depth and time).
-   `Data3D`: For 3D data, currently tailored to MOOSE simulation outputs which represent a 2D spatial field evolving over time.
-   `Geometry3D`: For 3D spatial path data, like a wellbore trajectory, which may not have a time component.
-   `TensorProcessor`: For tensor data that evolves over time (e.g., a 3x3 stress tensor at each time step).

This structure ensures that methods are defined in the correct context. For example, `select_depth` makes sense for `Data2D` but not `Data1D`.

### 3. Inheritance for Specialization (Core vs. Specialized Classes)

Within each dimensional package, there is a clear pattern of inheritance:

-   **`core<Dim>.py`**: This file contains the base class for that dimension (e.g., `core1D.py` contains `Data1D`). The base class implements the vast majority of the functionality that is common to all data of that dimension (e.g., loading/saving, cropping, shifting, plotting, merging).
-   **Specialized Classes**: Other files in the package define classes that inherit from the base class (e.g., `Data1D_PumpingCurve(Data1D)`). These classes are used for two primary purposes:
    1.  To provide a clear, named container for a specific *type* of data source.
    2.  To add highly specialized methods that only make sense for that specific data type (e.g., `get_start_time` on a `Data1DPumpingCurve`).

**Developers should always inherit from a core class and avoid re-implementing common functionality.**

### 4. Immutable History and Data Provenance

**This is the most important principle.** Every `analyzer` object has a `self.history` attribute, which is an instance of `InfoManagementSystem`. Every method that modifies the object's data or metadata **must** log its action.

-   **Why?** This provides a complete, traceable audit trail of how the data reached its current state. It is invaluable for debugging, ensuring reproducibility, and understanding the results of a complex analysis workflow.
-   **How?** Simply call `self.history.add_record("Your descriptive message here.", level="INFO")`.

### 5. Standardized I/O with `.npz`

While the `fiberis.io` module is responsible for reading dozens of raw, complex data formats, the `analyzer` module has a single, standardized format for saving and loading its objects: the NumPy `.npz` format.

-   The `savez()` method saves the processed state of an object (`data`, `taxis`, `start_time`, etc.) to a `.npz` file.
-   The `load_npz()` method perfectly reconstructs the object from that file.
-   This creates a clean separation. You use `io` once to get the data into an `analyzer` object, then all subsequent saving and loading of processed checkpoints uses the fast and simple `savez`/`load_npz` methods.

# Developer's Guide & Design Patterns

### Creating a New Specialized Data Class

Follow these steps to add support for a new type of data.

**Scenario**: We want to create a class for a specific type of temperature log from a new device. The data is a 1D time-series.

1.  **Choose the Dimension**: It's a 1D time-series, so we will build on `Data1D`.

2.  **Create the File**: In the `src/fiberis/analyzer/Data1D/` directory, create a new file named `Data1D_TempLog.py`.

3.  **Define the Class**: In the new file, define your class, inheriting from `core1D.Data1D`.

    ```python
    # src/fiberis/analyzer/Data1D/Data1D_TempLog.py
    from fiberis.analyzer.Data1D import core1D
    import numpy as np

    class Data1DTempLog(core1D.Data1D):
        """
        Represents a temperature log from the new device.
        Inherits all common functionality from Data1D.
        """
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.history.add_record("Initialized as Data1DTempLog object.", level="INFO")

        def calculate_average_temp(self, unit='C') -> float:
            """
            Calculates the average temperature over the current data range.

            Args:
                unit (str): The unit to return, 'C' for Celsius or 'F' for Fahrenheit.

            Returns:
                float: The average temperature.
            """
            if self.data is None or self.data.size == 0:
                self.history.add_record("Cannot calculate average temp, data is empty.", level="WARNING")
                return 0.0

            avg_c = np.mean(self.data)
            self.history.add_record(f"Calculated average temperature: {avg_c:.2f} C.", level="INFO")

            if unit == 'F':
                avg_f = (avg_c * 9/5) + 32
                self.history.add_record(f"Converted average to {avg_f:.2f} F.", level="INFO")
                return avg_f
            elif unit != 'C':
                raise ValueError("Unit must be 'C' or 'F'.")

            return avg_c
    ```

### Implementing Methods: Best Practices

#### 1. Operations are In-Place

By default, methods that manipulate data (e.g., `crop`, `shift`, `select_depth`) modify the object directly. This is a deliberate design choice for memory efficiency and to facilitate a fluent, chainable style of programming.

If you need to preserve the original object, the user is expected to create a copy first.

```python
# GOOD: The crop method modifies the object in-place
original_data.crop(start_time, end_time)

# GOOD: If the original is needed, the user makes a copy
processed_data = original_data.copy()
processed_data.crop(start_time, end_time)
# Now original_data is unchanged, and processed_data is cropped.
```

#### 2. Always Log to History

Every public method that changes or analyzes the data must log its actions.

```python
def my_new_method(self, some_param: int):
    # --- Start of method ---
    self.history.add_record(f"Running my_new_method with param: {some_param}", level="INFO")

    # --- Guard clauses and validation ---
    if self.data is None:
        self.history.add_record("Cannot run, data is not loaded.", level="ERROR")
        raise ValueError("Data is not loaded.")

    # --- Main logic ---
    # ... do some work ...
    result = "some_result"
    num_points_affected = 10

    # --- End of method ---
    self.history.add_record(f"Method complete. Affected {num_points_affected} points. Result: {result}", level="INFO")
```

#### 3. Use Consistent Naming

-   **Attributes**: `data`, `taxis`, `daxis`, `zaxis`, `start_time`, `name`, `history`.
-   **Methods**: `load_npz`, `savez`, `plot`, `copy`, `shift`, `crop`.

# Example Workflow

This example demonstrates how the components work together.

```python
import datetime
import numpy as np
from fiberis.analyzer.Data1D import Data1DGauge

# 1. Initialization
# In a real scenario, a reader from fiberis.io would create and populate this object.
# For this example, we'll create it manually.
start = datetime.datetime(2025, 8, 15, 12, 0, 0)
time_axis = np.arange(0, 60, 1) # 60 seconds of data
pressure_data = 1000 + np.random.randn(60) # Add an obvious spike
pressure_data[30] = 1500 

gauge = Data1DGauge(data=pressure_data, taxis=time_axis, start_time=start, name="WellheadPressure")

# 2. Analysis and Processing (chain of in-place operations)
# The user wants to focus on the first 45 seconds and remove the spike.

# Create a copy to keep the original safe
processed_gauge = gauge.copy()

# Crop the data
crop_end_time = start + datetime.timedelta(seconds=45)
processed_gauge.crop(start, crop_end_time)

# Remove the abnormal spike
processed_gauge.remove_abnormal_data(threshold=100, method='mean')

# Shift the data forward by 5 minutes
processed_gauge.shift(shift_val=300) # shift by 300 seconds

# 3. Visualization
# The plot method uses the object's internal state
processed_gauge.plot(title="Cleaned Pressure Data", use_timestamp=True)

# 4. Audit and Review
# The history tells us exactly what we did
print("--- Processing History ---")
processed_gauge.history.print_records()

# 5. Save for Later
# Save the processed object to the standard format
processed_gauge.savez("processed_wellhead_pressure.npz")

# Later, you can perfectly restore it
reloaded_gauge = Data1DGauge()
reloaded_gauge.load_npz("processed_wellhead_pressure.npz")
print("\n--- Reloaded Object ---")
print(f"Name: {reloaded_gauge.name}")
print(f"New Start Time: {reloaded_gauge.start_time}")
```