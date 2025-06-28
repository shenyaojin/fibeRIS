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

The `fiberis.io` package handles reading and writing standard data that can be handled by `fiberis.analyzer`.

## 2. Data Analysis (`fiberis.analyzer`)

This package provides classes and methods for analyzing various types of fiber optic and related reservoir data.

- **1D Data Analysis (`fiberis.analyzer.Data1D`)**:
  - `core1D.Data1D`: The base class for handling 1D datasets. It includes functionalities for loading data from `.npz` files, time-based cropping (`crop`, `select_time`), time shifting (`shift`), data retrieval (`get_value_by_time`), time calculations (`calculate_time`), plotting, merging datasets (`right_merge`), and basic signal processing like removing abnormal data points (`remove_abnormal_data`) and interpolation (`interpolate`). 
  - `Data1D_Gauge.Data1DGauge`: A specialized class for 1D gauge data, inheriting from `Data1D`. 
  - `Data1D_PumpingCurve.Data1DPumpingCurve`: Tailored for 1D pumping curve data, providing methods to determine start and end times. 
- **2D Data Analysis (`fiberis.analyzer.Data2D`)**:
  - `core2D.Data2D`: The base class for 2D datasets (typically distance vs. time). It supports loading, setting axes (`taxis`, `daxis`), time and depth selection/cropping (`select_time`, `select_depth`), plotting (imshow, pcolormesh with timestamp support), merging (`right_merge`), and signal processing (e.g., `apply_lowpass_filter`). 
  - `Data2D_XT_DSS.DSS2D`: Specialized for 2D DSS (Distributed Strain Sensing) data, often represented as XT plots (distance vs. time). 
- **3D Data Analysis (`fiberis.analyzer.Data3D`)**:
  - `core3D.py`: Currently a placeholder for future 3D time-discretized data analysis. 
- **3D Geometry Analysis (`fiberis.analyzer.Geometry3D`)**:
  - `coreG3D.DataG3D`: Base class for handling 3D geometrical data (e.g., wellbore trajectories), with methods like `calculate_md` (measured depth). 
  - `DataG3D_md.G3DMeasuredDepth`: Specialized class for measured depth type 3D geometry data. 

## 3. Simulation (`fiberis.simulator`)

This package provides tools for simulating reservoir phenomena, currently focusing on 1D pressure diffusion.

- **Core Simulation (`fiberis.simulator.core`)**:
  - `pds.PDS1D_SingleSource` & `pds.PDS1D_MultiSource`: Classes for setting up and solving 1D pressure diffusion problems with single or multiple source terms. They manage mesh, source terms, boundary conditions, initial conditions, and diffusivity. 
  - `bcs.BoundaryCondition`: Defines boundary conditions (Dirichlet, Neumann) for the PDS simulator. 
- **Solvers (`fiberis.simulator.solver`)**:
  - `matbuilder.py`: Builds the matrices (A, b) for the 1D diffusion problem. 
  - `PDESolver_IMP.solver_implicit`: Solves the PDE system using implicit methods (leveraging SciPy/NumPy). 
  - `PDESolver_EXP.solver_explicit`: Solves using an explicit iterative method (Jacobi). 
- **Optimizer (`fiberis.simulator.optimizer`)**:
  - `tso.time_sampling_optimizer`: Implements dynamic time step adjustment based on solution error for the PDS simulator. 

## 4. MOOSE Framework Extension (`fiberis.moose`)

This sub-package provides a comprehensive suite of tools for interacting with the MOOSE framework, enabling programmatic control over simulation workflows. For a detailed overview, refer to the [fibeRIS MOOSE Extension README](src/fiberis/moose/README.md). 

- **Configuration (`config.py`)**: Defines Python classes (`HydraulicFractureConfig`, `SRVConfig`) to specify parameters for hydraulic fractures and stimulated reservoir volumes. 
- **Input File Generation (`input_generator.py`)**: Generates MOOSE input files (`.i`) from Python dictionary configurations. Includes `MooseBlock` base class. 
- **Model Building (`model_builder.py`)**: A higher-level API to construct MOOSE input files using the configuration objects, simplifying the definition of physical features and mesh operations. 
- **Input File Editing (`input_editor.py`)**: Allows reading, modifying, and writing existing MOOSE input files using `pyhit` and `moosetree`. 
- **Simulation Execution (`runner.py`)**: Programmatically runs MOOSE simulations, captures output, and manages the execution process, including MPI support. 
- **Post-processing (`postprocessor.py`)**: Reads MOOSE output files (primarily Exodus `.e` files) using `meshio` and extracts data for analysis, including nodal/cell variables and time-dependent data. 

## 5. Utilities (`fiberis.utils`)

A collection of helper functions supporting various operations across the `fibeRIS` package.

- `history_utils.py`: A simple logging system (`InfoManagementSystem`) for recording operations with timestamps. 
- `io_utils.py`: Utilities for file I/O, such as `read_h5` for reading HDF5 files. 
- `mesh_utils.py`: Functions for mesh operations like refining a mesh (`refine_mesh`) and locating points (`locate`). 
- `signal_utils.py`: A comprehensive set of signal processing tools, including Butterworth filters (bandpass, lowpass, highpass), FFT-based amplitude spectrum calculation, NaN filling, time difference calculation, interpolation matrix generation, curve smoothing, RMS calculation, MATLAB datenum conversion, cross-correlation, and timestamp parsing. 
- `syn_utils.py`: Utilities for generating synthetic discrete time series data. 
- `viz_utils.py`: Placeholder for enhanced visualization tools, particularly for multi-stage hydraulic fracturing. 

# Examples

The `examples` directory contains scripts and notebooks demonstrating various functionalities of `fibeRIS`:

- `101r_real_DASdata_viz_hwell.py`: Visualizing real-world DAS data from a horizontal well, integrating pumping data and gauge measurements. 
- `102r_IOstandard_example.ipynb` and `102rp_IOstandard_example.py`: Showcasing the data input/output standards and reader functionalities.  

# Future Work

- [ ] Fully implement `reader_mariner_rfs.py` for Mariner RFS data.
- [ ] Expand 3D data processing capabilities in `fiberis.analyzer.Data3D`.
- [ ] Further develop visualization tools in `fiberis.utils.viz_utils.py`.
- [ ] Continuously refine and test the 1D PDS simulator, addressing any known issues.

# License

This project is licensed under the WTFPL – Do What the Fuck You Want to Public License. See the LICENSE file for details.
