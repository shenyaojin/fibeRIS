# fibeRIS

**fibeRIS: Fiber Optic Reservoir Integrated Simulator**

`fibeRIS` is a Python-based toolkit for the analysis, simulation, and management of data relevant to reservoir engineering, with a particular focus on Distributed Fiber Optic Sensing (DFOS) data. This project provides a suite of modules for handling multi-dimensional datasets, performing signal processing, simulating pressure diffusion, and programmatically controlling the [MOOSE (Multiphysics Object-Oriented Simulation Environment)](https://mooseframework.inl.gov/) framework.

## Installation

To install `fibeRIS` in editable mode, which is recommended during its ongoing development, clone the repository and run:

```bash
pip install -e .
```

If you want to use `moose` extension, please check this [Note](https://shenyaojin.github.io/theory/fiberis-setup/).

## Core Packages

`fibeRIS` is organized into several key packages that work together to support a comprehensive data processing and simulation workflow.

### 1. `fiberis.io`

The data ingestion engine of the toolkit. This package provides a collection of readers to load data from various raw or proprietary file formats and convert them into a standardized structure for analysis.

**[>> Learn more in the `fiberis.io` README](./src/fiberis/io/README.md)**

### 2. `fiberis.analyzer`

The core data analysis package. It contains classes and methods for handling various types of fiber optic and related reservoir data, organized by data dimensionality.

-   **`Data1D`**: For 1D time-series data like gauge pressure/temperature and pumping curves.
-   **`Data2D`**: For 2D datasets like DAS/DSS waterfall plots (distance vs. time).
-   **`Data3D`**: For 3D datasets, particularly for post-processing spatial data over time from MOOSE simulations.
-   **`Geometry3D`**: For handling 3D spatial path data, such as wellbore trajectories.
-   **`TensorProcessor`**: For handling time-dependent tensor data (e.g., stress/strain tensors).

**[>> Learn more in the `fiberis.analyzer` READMEs](./src/fiberis/analyzer/)**

### 3. `fiberis.simulator`

A lightweight, 1D finite-difference simulator for modeling pressure diffusion in porous media. It includes tools for setting up the simulation domain, defining boundary conditions and sources, and solving the system with either implicit or explicit methods. It also features an optimizer for adaptive time-stepping.

**[>> Learn more in the `fiberis.simulator` README](./src/fiberis/simulator/README.md)**

### 4. `fiberis.moose`

A comprehensive Python interface for the MOOSE framework. This package allows for the complete automation of the simulation workflow, from building input files programmatically to execution and post-processing.

-   **`config`**: A set of intuitive Python classes to represent physical and numerical concepts in MOOSE.
-   **`model_builder`**: A high-level, fluent API for constructing complex MOOSE input files from the ground up, including advanced features like stitched mesh generation for multi-fracture systems.
-   **`runner`**: A tool to programmatically run MOOSE simulations, including in parallel.
-   **`postprocessor`**: Utilities to read and analyze MOOSE output files (both Exodus `.e` and CSV formats).
-   **`input_editor`**: A tool to load, programmatically modify, and save existing MOOSE input files.

**[>> Learn more in the `fiberis.moose` README](./src/fiberis/moose/README.md)**

### 5. `fiberis.utils`

A collection of essential, reusable helper functions that support operations across the entire `fibeRIS` toolkit. This includes a rich `signal_utils` module for filtering and correlation, a history/logging system, mesh utilities, and more.

**[>> Learn more in the `fiberis.utils` README](./src/fiberis/utils/README.md)**

## Examples

The `examples` directory contains scripts and notebooks demonstrating various functionalities of `fibeRIS`, from visualizing real-world DAS data to showcasing the standard data I/O workflow.

## Future Work

-   Complete the implementation of the `reader_mariner_rfs.py` for RFS data.
-   Expand 3D data processing and visualization capabilities.
-   Continuously refine and validate the 1D PDS simulator.

## License

This project is licensed under the WTFPL – Do What the Fuck You Want to Public License. See the `LICENSE` file for details.