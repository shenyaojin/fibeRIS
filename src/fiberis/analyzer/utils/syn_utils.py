# Synthetic data generation utilities
# Shenyao Jin, 2025-02

import numpy as np
import datetime


def gen_discrete_time_series(s=100, **kwargs):
    """
    Generates a discrete time series with optional random or user-defined data.

    Parameters:
    s (int): Size of the time series (default is 100).
    random (bool, optional): If True, generates random data. If False, expects 'x' parameter (default is True).
    start_time (datetime.datetime, optional): Starting time of the time series. If not provided, defaults to the current time.
    seed (int, optional): Seed for the random number generator, used only when random=True.
    x (array-like, optional): User-defined data for the time series, required if random=False.
    filename (str, optional): Name of the file to save the time series data. The file will be saved in .npz format.

    Returns:
    tuple: A tuple containing the time axis (t), data (x), and the start time.
    """
    # Extract parameters from kwargs
    random = kwargs.get('random', True)  # Determine whether to generate random data or use provided data
    start_time = kwargs.get('start_time', None)  # Optional start time parameter

    # Validate the start_time parameter if provided
    if start_time is not None and not isinstance(start_time, datetime.datetime):
        raise ValueError("The 'start_time' parameter must be of type datetime.datetime.")

    if random:
        # Handle random data generation mode
        seed = kwargs.get('seed', None)
        if seed is not None:
            np.random.seed(seed)  # Set the seed if provided for reproducible random numbers
        t = np.arange(s)  # Generate a time axis with 's' points
        x = np.random.rand(s)  # Generate random data of the same size as the time axis
        if start_time is None:
            start_time = datetime.datetime.now()  # Use the current time if no start time is provided
    else:
        # Handle user-defined data mode
        t = np.arange(s)  # Generate a time axis with 's' points
        if 'x' in kwargs:
            x = kwargs['x']  # Retrieve the user-defined data
            if len(x) != s:
                raise ValueError("The 'x' parameter must have the same size as 't'.")  # Validate data size
        else:
            raise ValueError("The 'x' parameter is required when random is set to False.")  # Error if no data provided

    # Save the data to a file if a filename is provided
    filename = kwargs.get('filename', None)
    if filename is not None:
        np.savez(filename, taxis=t, data=x,
                 start_time=start_time)  # Save time axis, data, and start time in .npz format

    # Return the generated time series data and metadata
    return t, x, start_time