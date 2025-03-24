# fibeRIS
DFOS analyzer/simulator designed for reservoir engineering. 

I'll finish this MD file to illustrate what this package can do in the future.

# Installation

Use editable mode to install the package, for it's still under development. I'll upload it to PyPI when I think it's complete.

```bash
pip install -e .
```

# Workflow
## Data processing
The **fiberis** only accepts the data packed with the npz file with specific variables for I/O performance.

All the datasets are recognized as ref axis(`xaxis`, `taxis`, ...) and data(keyname `data`). 

# Future work

- [ ] Add HFTS2 IO
- [ ] Add 3D IO and processing
- [ ] Implement more signal processing methods in 2D
- [ ] Rewrite the simulation part -> There is a fatal bug in 1D single source simulator. Need to fix it ASAP!!!