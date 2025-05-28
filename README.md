# fibeRIS

**fibeRIS: Distributed Fiber Optic Sensing (DFOS) Analyzer and Simulator for Reservoir Engineering**

`fibeRIS` is a Python-based toolkit designed for the analysis, simulation, and management of data relevant to reservoir engineering, with a particular focus on Distributed Fiber Optic Sensing (DFOS) data. This project provides modules for handling multi-dimensional datasets, performing signal processing, simulating pressure diffusion, and interacting with the MOOSE (Multiphysics Object-Oriented Simulation Environment) framework.

# Installation

To install `fibeRIS` in editable mode (recommended during its ongoing development):

```
pip install -e .
```

# Core Features & Workflow

`fibeRIS` is organized into several key packages to support a comprehensive data processing and simulation workflow:

## 1. Data Input/Output (`fiberis.io`)

The `fiberis.io` package handles reading and writing data. `fibeRIS` primarily utilizes the `.npz` file format with specific variable naming conventions for optimized I/O operations. Datasets are typically structured with reference axes (e.g., `taxis` for time, `daxis` for depth/distance) and the primary data array (named `data`).

- **Packing Standard**: For detailed guidelines on preparing your data for use with `fibeRIS`, please refer to the [Packing Standard documentation](docs/packing_starndard.md). [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/docs/packing_starndard.md]
- **Core I/O Class**: `fiberis.io.core.DataIO` serves as an abstract base class for all data readers and writers. [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/src/fiberis/io/core.py]
- **Supported Data Readers**:
  - `reader_mariner_gauge1d`: For Mariner 1D gauge data. [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/src/fiberis/io/reader_mariner_gauge1d.py]
  - `reader_mariner_pp1d`: For Mariner 1D pumping data. [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/src/fiberis/io/reader_mariner_pp1d.py]
  - `reader_mariner_das2d`: For Mariner 2D DAS (Distributed Acoustic Sensing) data. [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/src/fiberis/io/reader_mariner_das2d.py]
  - `reader_mariner_3d`: For Mariner 3D well geometry data. [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/src/fiberis/io/reader_mariner_3d.py]
  - `reader_hfts2_h5`: For HFTS2 (Hydraulic Fracturing Test Site 2) data from HDF5 files. [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/src/fiberis/io/reader_hfts2_h5.py]
  - `reader_MOOSEcsv_pp1d`: For 1D post-processed data from MOOSE-generated CSV files. [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/src/fiberis/io/reader_MOOSEcsv_pp1d.py]
  - `reader_mariner_rfs`: For Mariner 2D RFS (Rayleigh Frequency Shift) data (implementation TBD). [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/src/fiberis/io/reader_mariner_rfs.py]
- **I/O Examples**:
  - See `examples/102r_IOstandard_example.ipynb` [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/examples/102r_IOstandard_example.ipynb] and `examples/102rp_IOstandard_example.py` [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/examples/102rp_IOstandard_example.py].

## 2. Data Analysis (`fiberis.analyzer`)

This package provides classes and methods for analyzing various types of fiber optic and related reservoir data.

- **1D Data Analysis (`fiberis.analyzer.Data1D`)**:
  - `core1D.Data1D`: The base class for handling 1D datasets. It includes functionalities for loading data from `.npz` files, time-based cropping (`crop`, `select_time`), time shifting (`shift`), data retrieval (`get_value_by_time`), time calculations (`calculate_time`), plotting, merging datasets (`right_merge`), and basic signal processing like removing abnormal data points (`remove_abnormal_data`) and interpolation (`interpolate`). [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/src/fiberis/analyzer/Data1D/core1D.py]
  - `Data1D_Gauge.Data1DGauge`: A specialized class for 1D gauge data, inheriting from `Data1D`. [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/src/fiberis/analyzer/Data1D/Data1D_Gauge.py]
  - `Data1D_PumpingCurve.Data1DPumpingCurve`: Tailored for 1D pumping curve data, providing methods to determine start and end times. [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/src/fiberis/analyzer/Data1D/Data1D_PumpingCurve.py]
- **2D Data Analysis (`fiberis.analyzer.Data2D`)**:
  - `core2D.Data2D`: The base class for 2D datasets (typically distance vs. time). It supports loading, setting axes (`taxis`, `daxis`), time and depth selection/cropping (`select_time`, `select_depth`), plotting (imshow, pcolormesh with timestamp support), merging (`right_merge`), and signal processing (e.g., `apply_lowpass_filter`). [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/src/fiberis/analyzer/Data2D/core2D.py]
  - `Data2D_XT_DSS.DSS2D`: Specialized for 2D DSS (Distributed Strain Sensing) data, often represented as XT plots (distance vs. time). [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/src/fiberis/analyzer/Data2D/Data2D_XT_DSS.py]
- **3D Data Analysis (`fiberis.analyzer.Data3D`)**:
  - `core3D.py`: Currently a placeholder for future 3D time-discretized data analysis. [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/src/fiberis/analyzer/Data3D/core3D.py]
- **3D Geometry Analysis (`fiberis.analyzer.Geometry3D`)**:
  - `coreG3D.DataG3D`: Base class for handling 3D geometrical data (e.g., wellbore trajectories), with methods like `calculate_md` (measured depth). [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/src/fiberis/analyzer/Geometry3D/coreG3D.py]
  - `DataG3D_md.G3DMeasuredDepth`: Specialized class for measured depth type 3D geometry data. [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/src/fiberis/analyzer/Geometry3D/DataG3D_md.py]

## 3. Simulation (`fiberis.simulator`)

This package provides tools for simulating reservoir phenomena, currently focusing on 1D pressure diffusion.

- **Core Simulation (`fiberis.simulator.core`)**:
  - `pds.PDS1D_SingleSource` & `pds.PDS1D_MultiSource`: Classes for setting up and solving 1D pressure diffusion problems with single or multiple source terms. They manage mesh, source terms, boundary conditions, initial conditions, and diffusivity. [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/src/fiberis/simulator/core/pds.py]
  - `bcs.BoundaryCondition`: Defines boundary conditions (Dirichlet, Neumann) for the PDS simulator. [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/src/fiberis/simulator/core/bcs.py]
- **Solvers (`fiberis.simulator.solver`)**:
  - `matbuilder.py`: Builds the matrices (A, b) for the 1D diffusion problem. [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/src/fiberis/simulator/solver/matbuilder.py]
  - `PDESolver_IMP.solver_implicit`: Solves the PDE system using implicit methods (leveraging SciPy/NumPy). [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/src/fiberis/simulator/solver/PDESolver_IMP.py]
  - `PDESolver_EXP.solver_explicit`: Solves using an explicit iterative method (Jacobi). [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/src/fiberis/simulator/solver/PDESolver_EXP.py]
- **Optimizer (`fiberis.simulator.optimizer`)**:
  - `tso.time_sampling_optimizer`: Implements dynamic time step adjustment based on solution error for the PDS simulator. [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/src/fiberis/simulator/optimizer/tso.py]

## 4. MOOSE Framework Extension (`fiberis.moose`)

This sub-package provides a comprehensive suite of tools for interacting with the MOOSE framework, enabling programmatic control over simulation workflows. For a detailed overview, refer to the [fibeRIS MOOSE Extension README](src/fiberis/moose/README.md). [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/src/fiberis/moose/README.md]

- **Configuration (`config.py`)**: Defines Python classes (`HydraulicFractureConfig`, `SRVConfig`) to specify parameters for hydraulic fractures and stimulated reservoir volumes. [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/src/fiberis/moose/config.py]
- **Input File Generation (`input_generator.py`)**: Generates MOOSE input files (`.i`) from Python dictionary configurations. Includes `MooseBlock` base class. [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/src/fiberis/moose/input_generator.py]
- **Model Building (`model_builder.py`)**: A higher-level API to construct MOOSE input files using the configuration objects, simplifying the definition of physical features and mesh operations. [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/src/fiberis/moose/model_builder.py]
- **Input File Editing (`input_editor.py`)**: Allows reading, modifying, and writing existing MOOSE input files using `pyhit` and `moosetree`. [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/src/fiberis/moose/input_editor.py]
- **Simulation Execution (`runner.py`)**: Programmatically runs MOOSE simulations, captures output, and manages the execution process, including MPI support. [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/src/fiberis/moose/runner.py]
- **Post-processing (`postprocessor.py`)**: Reads MOOSE output files (primarily Exodus `.e` files) using `meshio` and extracts data for analysis, including nodal/cell variables and time-dependent data. [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/src/fiberis/moose/postprocessor.py]

## 5. Utilities (`fiberis.utils`)

A collection of helper functions supporting various operations across the `fibeRIS` package.

- `history_utils.py`: A simple logging system (`InfoManagementSystem`) for recording operations with timestamps. [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/src/fiberis/utils/history_utils.py]
- `io_utils.py`: Utilities for file I/O, such as `read_h5` for reading HDF5 files. [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/src/fiberis/utils/io_utils.py]
- `mesh_utils.py`: Functions for mesh operations like refining a mesh (`refine_mesh`) and locating points (`locate`). [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/src/fiberis/utils/mesh_utils.py]
- `signal_utils.py`: A comprehensive set of signal processing tools, including Butterworth filters (bandpass, lowpass, highpass), FFT-based amplitude spectrum calculation, NaN filling, time difference calculation, interpolation matrix generation, curve smoothing, RMS calculation, MATLAB datenum conversion, cross-correlation, and timestamp parsing. [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/src/fiberis/utils/signal_utils.py]
- `syn_utils.py`: Utilities for generating synthetic discrete time series data. [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/src/fiberis/utils/syn_utils.py]
- `viz_utils.py`: Placeholder for enhanced visualization tools, particularly for multi-stage hydraulic fracturing. [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/src/fiberis/utils/viz_utils.py]

# Examples

The `examples` directory contains scripts and notebooks demonstrating various functionalities of `fibeRIS`:

- `101r_real_DASdata_viz_hwell.py`: Visualizing real-world DAS data from a horizontal well, integrating pumping data and gauge measurements. [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/examples/101r_real_DASdata_viz_hwell.py]
- `102r_IOstandard_example.ipynb` and `102rp_IOstandard_example.py`: Showcasing the data input/output standards and reader functionalities. [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/examples/102r_IOstandard_example.ipynb] [cite: uploaded:shenyaojin/fiberis/fibeRIS-fb657b87e09e639b690d0ba5a727f42a3c87ca2e/examples/102rp_IOstandard_example.py]

# Future Work

- [ ] Fully implement `reader_mariner_rfs.py` for Mariner RFS data.
- [ ] Expand 3D data processing capabilities in `fiberis.analyzer.Data3D`.
- [ ] Further develop visualization tools in `fiberis.utils.viz_utils.py`.
- [ ] Continuously refine and test the 1D PDS simulator, addressing any known issues.

# License

This project is licensed under the WTFPL – Do What the Fuck You Want to Public License. See the LICENSE file for details.
