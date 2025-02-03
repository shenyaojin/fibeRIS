# Implicit solver for the Diffusion PDE; uses the Crank-Nicolson method temporarily.
# Solve the matrix for one time step and will be called in pds.py.

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la

def solver_implicit(A, b, **kwargs):
    """
    Solve the matrix using the implicit method.
    Args:
        A: Matrix A, 2D numpy array
        b: Matrix b, 1D numpy array
        **kwargs:
            solver: str, "scipy" or "numpy" or "direct"
    Returns:
        x: 1D numpy array
    """

    # Get the solver type if provided
    solver = kwargs.get('solver', 'scipy')

    # Solve the matrix
    if solver == 'scipy':
        x = spla.spsolve(A, b)
    elif solver == 'numpy':
        x = la.solve(A, b)
    elif solver == 'direct':
        x = np.linalg.solve(A, b) # Will be removed in the future
    else:
        raise ValueError('Invalid solver type.')

    return x