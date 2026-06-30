# Shared pytest fixtures and configuration for the fibeRIS test suite.
#
# These fixtures support the characterization tests that lock in the current
# observable behavior of the package, providing a safety net for the ongoing
# modernization/refactoring work.

import datetime
import os
import sys

import numpy as np
import pytest

# Ensure the package is importable even without an editable install.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Use a non-interactive matplotlib backend so plotting code never blocks tests.
import matplotlib

matplotlib.use("Agg")

# Absolute path to the bundled example datasets used as real-data fixtures.
EXAMPLES_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "examples", "data")
)


@pytest.fixture
def examples_data_dir():
    """Path to the repository's bundled example data directory."""
    return EXAMPLES_DATA_DIR


@pytest.fixture
def sample_start_time():
    """A fixed, deterministic start time used across tests."""
    return datetime.datetime(2023, 1, 1, 12, 0, 0)


@pytest.fixture
def sine_taxis_data():
    """A simple, deterministic 1D time series (taxis, data)."""
    taxis = np.linspace(0, 10, 11)
    data = np.sin(taxis)
    return taxis, data
