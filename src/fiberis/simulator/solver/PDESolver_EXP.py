# Shenyao Jin, shenyaojin@mins.edu, 02/05/2025
import numpy as np


def solver_explicit(A, b, tol=1e-10, max_iter=1000, **kwargs):
    """
    Solve the matrix using an explicit iterative method (Jacobi).

    Args:
        A: Matrix A, 2D numpy array (assumed to be diagonally dominant)
        b: Matrix b, 1D numpy array
        tol: Tolerance for convergence (default: 1e-10)
        max_iter: Maximum number of iterations (default: 1000)

    Returns:
        x: 1D numpy array, solution vector
    """
    # Initialize the solution vector with zeros
    x = np.zeros_like(b, dtype=np.float64)

    # Iterate using the Jacobi method
    for iteration in range(max_iter):
        x_new = np.zeros_like(x)

        # Update each element in the solution vector
        for i in range(A.shape[0]):
            sum_except_i = np.dot(A[i, :], x) - A[i, i] * x[i]
            x_new[i] = (b[i] - sum_except_i) / A[i, i]

        # Check for convergence
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            print(f'Converged in {iteration + 1} iterations.')
            return x_new

        x = x_new

    # If max iterations reached without convergence, raise a warning
    print('Warning: Maximum iterations reached without convergence.')
    return x