# Some utility functions for working with meshes.
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# setup mesh
import numpy as np

def refine_mesh(x, refine_range, refine_factor):
    """
    Refine the resolution within a specified range and return the new mesh.

    Parameters:
    - x (numpy.ndarray): Original mesh (1D array).
    - refine_range (tuple): Range to refine (start, end) based on coordinate values.
    - refine_factor (int): Refinement factor (how many times to increase resolution).

    Returns:
    - new_x (numpy.ndarray): Refined mesh array.

    Raises:
    - ValueError: If refine_range is invalid or refine_factor is less than 1.
    """
    # Validate inputs
    if not isinstance(x, np.ndarray):
        raise ValueError("Input x must be a numpy.ndarray representing a 1D array.")
    if len(refine_range) != 2 or refine_range[0] >= refine_range[1]:
        raise ValueError("refine_range must be a tuple of length 2 with start < end.")
    if refine_factor < 1:
        raise ValueError("refine_factor must be greater than or equal to 1.")

    # Find the indices corresponding to the refine range
    start, end = refine_range
    start_idx = np.searchsorted(x, start, side='left')  # Include start point
    end_idx = np.searchsorted(x, end, side='right')    # Include end point

    # Extract the coarse resolution parts
    coarse_region = np.concatenate((x[:start_idx], x[end_idx:]))

    # Generate refined points in the specified range
    refined_points = np.linspace(start, end, (end_idx - start_idx) * refine_factor + 1)

    # Merge and sort the mesh
    new_x = np.sort(np.concatenate((coarse_region, refined_points)))

    return new_x

def locate(x, frac_loc):
    """
    Locate the index and value in x closest to the target location frac_loc.

    Parameters:
    - x (numpy.ndarray): 1D array representing the mesh.
    - frac_loc (float): Target location to find the nearest point.

    Returns:
    - ind (int): Index of the closest value in x.
    - closest_value (float): Value in x closest to frac_loc.

    Raises:
    - ValueError: If x is not a numpy array or if it's empty.
    """
    # Validate inputs
    if not isinstance(x, np.ndarray):
        raise ValueError("Input x must be a numpy.ndarray.")
    if x.size == 0:
        raise ValueError("Input x cannot be an empty array.")
    if not np.isscalar(frac_loc):
        raise ValueError("frac_loc must be a scalar value.")

    # Compute the absolute differences between frac_loc and elements in x
    differences = np.abs(x - frac_loc)

    # Find the index of the minimum difference
    ind = np.argmin(differences)

    # Get the closest value
    closest_value = x[ind]

    return ind, closest_value