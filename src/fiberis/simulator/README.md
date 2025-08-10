# `fiberis.simulator`

This package provides a lightweight, one-dimensional finite-difference simulator for modeling pressure diffusion in porous media, such as a reservoir rock surrounding a wellbore. It is designed to quickly simulate pressure responses that might be observed with fiber optic sensing.

The simulator is organized into three main sub-packages: `core`, `solver`, and `optimizer`.

## `fiberis.simulator.core`

This package contains the main components for setting up a simulation problem.

-   **`pds.py` (`PDS1D_SingleSource`, `PDS1D_MultiSource`)**: These are the main simulation classes.
    -   `PDS1D_SingleSource`: Sets up a 1D pressure diffusion problem with a single pressure source.
    -   `PDS1D_MultiSource`: Extends the single-source class to handle multiple, independent pressure sources along the 1D domain.
    -   **Key Responsibilities**: These classes hold all the simulation parameters: the mesh, initial conditions, boundary conditions, material properties (diffusivity), and source term definitions. They contain the main `solve` loop that orchestrates the simulation step-by-step.

-   **`bcs.py` (`BoundaryCondition`)**: A helper class used to define the boundary conditions (e.g., Dirichlet, Neumann) that are applied at the ends of the 1D domain.

## `fiberis.simulator.solver`

This package contains the numerical engines that solve the system of linear equations at each time step.

-   **`matbuilder.py`**: This module is responsible for constructing the `A` matrix (representing the spatial discretization of the diffusion operator) and the `b` vector (representing the state at the previous time step plus source terms) for the linear system `Ax = b`. It can handle both single and multiple source terms.

-   **`PDESolver_IMP.py` (`solver_implicit`)**: An implicit solver that uses standard libraries like SciPy or NumPy to solve the linear system `Ax = b`. Implicit methods are generally stable even for large time steps.

-   **`PDESolver_EXP.py` (`solver_explicit`)**: An explicit solver that uses an iterative method (Jacobi) to solve the system. Explicit methods can be faster per time step but have stricter stability requirements, often needing very small time steps.

## `fiberis.simulator.optimizer`

This package provides tools for dynamically adjusting the simulation time step to improve efficiency and accuracy.

-   **`tso.py` (`time_sampling_optimizer`)**: This module implements a time-sampling optimizer. At each step, it compares the solution from a full time step with a solution from two half time steps. Based on the difference (error), it decides whether to accept the current step and adjusts the size of the next time step—making it larger if the error is small and smaller if the error is large.

## Workflow

1.  Instantiate a `PDS1D_SingleSource` or `PDS1D_MultiSource` object.
2.  Use the `set_*` methods (e.g., `set_mesh`, `set_source`, `set_diffusivity`, `set_bcs`) to define the problem. The source term is typically a `Data1D` object from the `fiberis.analyzer` package.
3.  Call the `solve` method, specifying the total time, initial time step (`dt`), and solver mode (`implicit` or `explicit`).
4.  Optionally, enable the `optimizer` to use adaptive time-stepping.
5.  After the simulation, the results are stored in the `snapshot` (a 2D array of pressure vs. time) and `taxis` attributes of the PDS object.
6.  The results can be visualized using the `plot_solution` method or packed into a standard `.npz` file using `pack_result` for analysis with the `Data2D` tools.