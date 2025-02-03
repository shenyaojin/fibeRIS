import numpy as np

# Should use core.pds to get the correct type of data

# Must use the pds object to get the data
def matrix_builder_1d_single_source(pds1d, dt):
    """
    Build the matrix for the 1D diffusion problem with a single source term.
    Args:
    pds: pds object
    dt: time step
    Returns:
    A: 2D numpy array
    b: 1D numpy array
    """
    # Calculate alpha; in this case, the dx is constant
    # dx = pds.mesh[1] - pds.mesh[0]
    # dx is not constant
    dx = np.diff(pds1d.mesh) # dx is an array of length nx-1
    diffusivity_eff = (2 * pds1d.diffusivity[:-1] * pds1d.diffusivity[1:] /
                      (pds1d.diffusivity[:-1] + pds1d.diffusivity[1:]))
    dxm = dx[:-1] # dxm is an array of length nx-2
    dxp = dx[1:] # dxp is an array of length nx-2

    alpha_l = diffusivity_eff[:-1] * dt / (dxm * (dxm + dxp) / 2) # alpha is an array of length nx-1
    alpha_r = diffusivity_eff[1:]  * dt / (dxp * (dxm + dxp) / 2) # alpha is an array of length nx-1

    nx = len(pds1d.mesh)

    # Define the matrix A
    A = np.zeros((nx, nx))

    # Define the vector b
    b = np.zeros(nx)

    # Fill the matrix A and vector b
    for iter in range(1, nx-1):
        A[iter, iter-1] = - alpha_l[iter-1]
        A[iter, iter] = 1 + alpha_l[iter-1] + alpha_r[iter-1]
        A[iter, iter+1] = - alpha_r[iter-1]

    # Fill the vector b, should be the snapshot of the previous time step. Use copy to avoid the reference
    b = pds1d.snapshot[-1].copy()

    # Apply the boundary conditions
    # Scan the left boundary
    if pds1d.lbc == 'Dirichlet':
        A[0, 0] = 1 # No need to initialize for the for loop above
        b[0] = 0
    elif pds1d.lbc == 'Neumann':
        A[0, 0] = -1
        A[0, 1] = 1
        b[0] = 0

    # Scan the right boundary
    if pds1d.rbc == 'Dirichlet':
        A[-1, -1] = 1
        b[-1] = 0
    elif pds1d.rbc == 'Neumann':
        A[-1, -1] = -1
        A[-1, -2] = 1
        b[-1] = 0

    # Apply the source term
    # Get the source value from PG data
    source_value = pds1d.source.get_value_by_time(pds1d.taxis[-1])
    A[pds1d.sourceidx, :] = 0 # Initialize the row
    A[pds1d.sourceidx, pds1d.sourceidx] = 1
    b[pds1d.sourceidx] = source_value

    return A, b

# Multiple sources
def matrix_builder_1d_multi_source(pds1d, t):
    """
    Build the matrix for the 1D diffusion problem with multiple source terms.
    Args:
        pds: pds object
        t: time

    Returns:
        A: 2D numpy array
        b: 1D numpy array
    """
    # Calculate alpha; in this case, the dx is constant (old)
    # dx = pds1d.mesh[1] - pds1d.mesh[0]
    # dx is not constant
    dx = np.diff(pds1d.mesh) # dx is an array of length nx-1
    diffusivity_eff = (2 * pds1d.diffusivity[:-1] * pds1d.diffusivity[1:] /
                        (pds1d.diffusivity[:-1] + pds1d.diffusivity[1:]))
    dxm = dx[:-1] # dxm is an array of length nx-2
    dxp = dx[1:] # dxp is an array of length nx-2

    alpha_l = diffusivity_eff[:-1] * t / (dxm * (dxm + dxp) / 2) # alpha is an array of length nx-1
    alpha_r = diffusivity_eff[1:]  * t / (dxp * (dxm + dxp) / 2) # alpha is an array of length nx-1

    nx = len(pds1d.mesh)

    # Define the matrix A
    A = np.zeros((nx, nx))

    # Define the vector b
    b = np.zeros(nx)

    # Fill the matrix A and vector b
    for iter in range(1, nx-1):
        A[iter, iter-1] = - alpha_l[iter-1]
        A[iter, iter] = 1 + alpha_l[iter-1] + alpha_r[iter-1]
        A[iter, iter+1] = - alpha_r[iter-1]

    # Fill the vector b, should be the snapshot of the previous time step. Use copy to avoid the reference
    b = pds1d.snapshot[-1].copy()

    # Apply the boundary conditions.
    # Scan the left boundary
    if pds1d.lbc == 'Dirichlet':
        A[0, 0] = 1 # No need to initialize for the for loop above
        b[0] = 0
    elif pds1d.lbc == 'Neumann':
        A[0, 0] = -1
        A[0, 1] = 1
        b[0] = 0

    # Scan the right boundary
    if pds1d.rbc == 'Dirichlet':
        A[-1, -1] = 1
        b[-1] = 0
    elif pds1d.rbc == 'Neumann':
        A[-1, -1] = -1
        A[-1, -2] = 1
        b[-1] = 0

    # Apply the source term， multiple sources
    # Get the source value from PG data
    source_value_list = []
    for iter_pg_dataframe in pds1d.source:
        source_value = iter_pg_dataframe.get_value_by_time(pds1d.taxis[-1])
        source_value_list.append(source_value)

    flag = 0 # flag for the first source
    for iter_sourceidx in pds1d.sourceidx:
        A[iter_sourceidx, :] = 0
        A[iter_sourceidx, iter_sourceidx] = 1
        b[iter_sourceidx] = (
            source_value_list)[flag]
        flag += 1

    return A, b