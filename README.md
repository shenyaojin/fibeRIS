# fibeRIS

**fibeRIS: Fiber Optic Reservoir Integrated Simulator**

`fibeRIS` is a Python-based toolkit for the analysis, simulation, and management of data relevant to reservoir engineering, with a particular focus on Distributed Fiber Optic Sensing (DFOS) data. It is developed by Shenyao Jin for research purposes.

This project provides a suite of modules for handling multi-dimensional datasets, performing signal processing, simulating pressure diffusion, and programmatically controlling the [MOOSE (Multiphysics Object-Oriented Simulation Environment)](https://mooseframework.inl.gov/) framework.

## Core Modules

### Analyzer (`fiberis.analyzer`)
The core of the data processing capabilities, providing specialized classes for different data dimensions:
*   **Data1D**: Handling 1D time-series data (e.g., gauge pressure, pumping curves).
*   **Data2D**: Handling 2D spatiotemporal data (e.g., DAS waterfall plots).
*   **Data3D**: Handling 3D volumetric data or multi-variable datasets.
*   **Data1DG**: Handling 1D geometric/spatial data (e.g., depth profiles).
*   **Geometry3D**: Handling 3D wellbore trajectories and spatial geometries.
*   **TensorProcessor**: Handling tensor data (e.g., stress/strain tensors) over time.

### Utilities (`fiberis.utils`)
A collection of helper functions for:
*   **Signal Processing**: Filtering (Butterworth), spectral analysis, outlier removal.
*   **History Management**: Logging operations and tracking data lineage.
*   **Visualization**: Plotting tools for 1D, 2D, and 3D data.

### Simulation (`fiberis.simulator` & `fiberis.moose`)
*   **Simulator**: A lightweight, independent 1D simulator for quick pressure diffusion modeling.
*   **MOOSE Wrapper**: A programmatic interface to build, run, and analyze complex multiphysics simulations using the MOOSE framework.

## Installation

You can install `fibeRIS` using pip:

```bash
pip install fiberis
```

To install from source for development:

```bash
git clone https://github.com/shenyaojin/fibeRIS.git
cd fibeRIS
pip install -e .
```

## Testing

The repository includes a comprehensive test suite based on `pytest`. To run the tests:

1.  Install test dependencies:
    ```bash
    pip install pytest numpy matplotlib scipy pandas
    ```

2.  Run tests:
    ```bash
    pytest tests/
    ```

## License

This project is licensed under MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please contact Shenyao Jin at `shenyaojin@mines.edu`.
