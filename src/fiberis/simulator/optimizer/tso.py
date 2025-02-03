import numpy as np


def time_sampling_optimizer(pds1d, full_step_solution, half_step_solution, dt, tol=1e-3, *args, **kwargs):
    """
    Optimize the time sampling for the 1D pressure diffusion problem.
    Args:
        pds1d: the 1D pressure diffusion problem object
        full_step_solution: the solution at the full time step calculated by the solver
        half_step_solution: the solution at the half time step calculated by the solver
        tol: the tolerance for the optimization, default is 1e-3
        *args: additional arguments
        **kwargs: additional keyword arguments
    Returns:
        No return value. All the results will be stored/modifed in the pds1d object.
    """
    # Calculate the error
    error = np.linalg.norm(full_step_solution - half_step_solution) / np.linalg.norm(full_step_solution)

    # Update the time parameter
    updated_dt = adjust_dt(dt, error, **kwargs)

    # Decide whether to accept the snapshot
    if error <= tol:
        # The error is small enough, then accept it and append the new snapshot/taxis
        pds1d.snapshot.append(full_step_solution)
        pds1d.taxis.append(pds1d.taxis[-1] + dt)
        pds1d.record_log("Dynamic time sampling updated. dt = ", dt, "Error = ", error, "Next loop dt will be",
                         updated_dt, "\n")
    else:
        # The error is too large, then reject it and keep the old snapshot/taxis; update the time step
        pds1d.record_log("Dynamic time sampling rejected. dt = ", dt, "Next loop dt will be", updated_dt, "\n")
    return updated_dt

def adjust_dt(dt, error, **kwargs):
    safety_factor = kwargs.get('safety_factor', 0.9)
    p             = kwargs.get('p', 2)
    max_dt        = kwargs.get('max_dt', 40)
    min_dt        = kwargs.get('min_dt', 1e-4)
    tol           = kwargs.get('tol', 1e-3)

    if error < 1e-14:
        ratio = 1.0
    else:
        ratio = (tol / error) ** (1.0 / p)
    # Calculate the new dt
    dt_new = dt * safety_factor * ratio
    # Make sure the new dt is within the range
    dt_new = max(min_dt, min(dt_new, max_dt))
    return dt_new