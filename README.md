# fibeRIS
DFOS analyzer/simulator designed for reservoir engineering. 

I'll finish this MD file to illustrate what this package can do in the future.

# Installation

Use editable mode to install the package, for it's still under development.

```bash
pip install -e .
```

# Workflow
## Data processing
The **fiberis** only accepts the data packed with the npz file with specific variables. 

All the datasets are recognized as ref axis(`xaxis`, `taxis`, ...) and data(keyname `data`). 