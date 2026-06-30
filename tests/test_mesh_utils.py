# Characterization tests for fiberis.utils.mesh_utils.

import numpy as np
import numpy.testing as npt
import pytest

from fiberis.utils import mesh_utils as mu


# ---------------------------------------------------------------------------
# refine_mesh
# ---------------------------------------------------------------------------

def test_refine_mesh_golden():
    x = np.arange(0, 11, dtype=float)  # 0..10
    out = mu.refine_mesh(x, (3, 6), 2)
    expected = [
        0.0, 1.0, 2.0, 3.0,
        3.375, 3.75, 4.125, 4.5, 4.875, 5.25, 5.625, 6.0,
        7.0, 8.0, 9.0, 10.0,
    ]
    npt.assert_allclose(out, expected, atol=1e-9)


def test_refine_mesh_factor3():
    x = np.linspace(0, 10, 11)
    out = mu.refine_mesh(x, (2.0, 4.0), 3)
    assert len(out) == 18
    npt.assert_allclose(out[0], 0.0, atol=1e-9)
    npt.assert_allclose(out[-1], 10.0, atol=1e-9)
    # Refined section between 2 and 4 inclusive.
    assert np.all(np.diff(out) >= -1e-12)  # sorted ascending


def test_refine_mesh_sorted_and_includes_endpoints():
    x = np.linspace(0, 100, 21)
    out = mu.refine_mesh(x, (20, 40), 5)
    assert np.all(np.diff(out) >= 0)
    assert out[0] == x[0]
    assert out[-1] == x[-1]


def test_refine_mesh_non_ndarray_raises():
    with pytest.raises(ValueError):
        mu.refine_mesh([0, 1, 2], (0, 1), 2)


def test_refine_mesh_bad_range_raises():
    x = np.arange(10.0)
    with pytest.raises(ValueError):
        mu.refine_mesh(x, (5, 3), 2)  # start >= end
    with pytest.raises(ValueError):
        mu.refine_mesh(x, (1, 2, 3), 2)  # wrong length


def test_refine_mesh_bad_factor_raises():
    x = np.arange(10.0)
    with pytest.raises(ValueError):
        mu.refine_mesh(x, (1, 5), 0)


# ---------------------------------------------------------------------------
# locate
# ---------------------------------------------------------------------------

def test_locate_nearest():
    ind, val = mu.locate(np.array([0.0, 1.0, 2.0, 3.0, 4.0]), 2.6)
    assert ind == 3
    npt.assert_allclose(val, 3.0, atol=1e-12)


def test_locate_beyond_range_clamps_to_nearest():
    ind, val = mu.locate(np.array([10.0, 20.0, 30.0]), 100)
    assert ind == 2
    npt.assert_allclose(val, 30.0, atol=1e-12)


def test_locate_exact():
    ind, val = mu.locate(np.array([0.0, 5.0, 10.0]), 5.0)
    assert ind == 1
    assert val == 5.0


def test_locate_non_ndarray_raises():
    with pytest.raises(ValueError):
        mu.locate([1, 2, 3], 2)


def test_locate_empty_raises():
    with pytest.raises(ValueError):
        mu.locate(np.array([]), 1.0)


def test_locate_non_scalar_raises():
    with pytest.raises(ValueError):
        mu.locate(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
