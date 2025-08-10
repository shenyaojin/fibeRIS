# `fiberis.utils`

The `fiberis.utils` package is a collection of helper modules and functions that provide essential, reusable functionality across the entire `fibeRIS` toolkit.

## Modules

### `signal_utils.py`

This is a comprehensive signal processing library containing a wide array of functions for data manipulation and analysis.

-   **Filtering**: Butterworth filters for low-pass, high-pass, and band-pass applications (`lpfilter`, `hpfilter`, `bpfilter`).
-   **Spectral Analysis**: Amplitude spectrum calculation using FFT (`amp_spectrum`).
-   **Correlation**: Cross-correlation using FFT (`rfft_xcorr`) and a function to find the optimal lag between two signals (`xcor_match`).
-   **Interpolation and Smoothing**:
    -   Generate interpolation matrices (`get_interp_mat`).
    -   A robust curve smoothing algorithm with iterative outlier rejection (`get_smooth_curve`).
    -   Interpolate data based on `datetime` objects (`datetime_interp`).
-   **Data Cleaning**: Fill `NaN` values in an array (`fillnan`).
-   **Time Series Analysis**:
    -   A windowed cross-correlation approach to find time shifts between two signals (`timeshift_xcor`).
    -   Running average with proper edge handling (`running_average`).
-   **Miscellaneous**: Root-mean-square calculation (`rms`), phase wrapping (`phase_wrap`), and fast `datetime` parsing (`fetch_timestamp_fast`).

### `history_utils.py`

-   **`InfoManagementSystem`**: A simple but effective logging class used throughout `fibeRIS`. It allows objects to maintain a history of operations performed on them, complete with timestamps and severity levels (INFO, WARNING, ERROR).

### `mesh_utils.py`

-   **`refine_mesh`**: A function to increase the resolution of a 1D mesh within a specified range.
-   **`locate`**: Finds the index of the point in a mesh that is closest to a given coordinate.

### `io_utils.py`

-   **`read_h5`**: A specific utility for reading data from HDF5 (`.h5`) files, particularly those from OptaSense DAS systems.

### `syn_utils.py`

-   **`gen_discrete_time_series`**: A utility for generating synthetic 1D time-series data, which is useful for testing and examples.

### `viz_utils.py`

A placeholder for future specialized visualization tools, potentially for more complex scenarios like multi-stage hydraulic fracturing.
