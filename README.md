# fibeRIS

**fibeRIS: Fiber Optic Reservoir Integrated Simulator**

`fibeRIS` is a Python-based toolkit for the analysis, simulation, and management of data relevant to reservoir engineering, with a particular focus on Distributed Fiber Optic Sensing (DFOS) data. It is developed by me and (currently) only serves for my research purpose.
I tried to make it compatible to unix-like systems, but I can't promise it could work on your machine.

This project provides a suite of modules for handling multi-dimensional datasets, performing signal processing, simulating pressure diffusion, and programmatically controlling the [MOOSE (Multiphysics Object-Oriented Simulation Environment)](https://mooseframework.inl.gov/) framework.

## Functions of fiberis

### Data management
TBD
### Data cleansing
TBD
### Data processing
TBD

### Simulation toolkits

`fiberis` provides two simulators, one is `fiberis.simulator` and another one is `fiberis.moose`. The difference between them is that the `simulator` is a light-weighted, independent 1d simulator, which can be used to quickly simulate the pressure diffusion along the fiber optic cable. 
The `moose` module is a wrapper of MOOSE framework, which can be used to build much more complex models and perform multiphysics simulations. Both simulators can be used to simulate the pressure diffusion along the fiber optic cable. And more importantly, the `moose` module can be used to interpret the strain response from the fiber optic cable using 
the model called HMM (Hydro-Mechanical Model).

See [This notebook](examples/scripts/simulator/moose_simulator.ipynb) to learn how to use the `moose` module to build a MOOSE model and simulate the pressure diffusion along the fiber optic cable.

## Installation

You can install `fibeRIS` using pip:

```bash
pip install fiberis
```

This GitHub repository is the lastest development version, while you can also install the stable version from PyPI. using the command above.

## License

This project is licensed under the WTFPL – Do What the Fuck You Want to Public License. See the `LICENSE` file for details.