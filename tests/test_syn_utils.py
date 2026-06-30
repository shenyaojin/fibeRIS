# Characterization tests for fiberis.utils.syn_utils.gen_discrete_time_series.

import datetime

import numpy as np
import numpy.testing as npt
import pytest

from fiberis.utils import syn_utils as syn


# ---------------------------------------------------------------------------
# Random mode (seeded -> reproducible)
# ---------------------------------------------------------------------------

def test_random_seeded_golden():
    t, x, st = syn.gen_discrete_time_series(
        s=5, random=True, seed=123, start_time=datetime.datetime(2023, 1, 1)
    )
    npt.assert_array_equal(t, [0, 1, 2, 3, 4])
    npt.assert_allclose(
        x,
        [0.69646919, 0.28613933, 0.22685145, 0.55131477, 0.71946897],
        atol=1e-8,
    )
    assert st == datetime.datetime(2023, 1, 1)


def test_random_default_time_axis_is_arange():
    t, x, _ = syn.gen_discrete_time_series(s=4, random=True, seed=0)
    npt.assert_array_equal(t, [0, 1, 2, 3])
    assert len(x) == 4


def test_random_custom_time_axis():
    t, x, _ = syn.gen_discrete_time_series(
        s=4, random=True, seed=0, t=np.array([10.0, 20.0, 30.0, 40.0])
    )
    npt.assert_array_equal(t, [10.0, 20.0, 30.0, 40.0])
    npt.assert_allclose(
        x, [0.5488135, 0.71518937, 0.60276338, 0.54488318], atol=1e-8
    )


def test_random_default_start_time_is_now():
    before = datetime.datetime.now()
    _, _, st = syn.gen_discrete_time_series(s=3, random=True, seed=1)
    after = datetime.datetime.now()
    assert before <= st <= after


# ---------------------------------------------------------------------------
# User-defined data mode
# ---------------------------------------------------------------------------

def test_user_defined_data():
    t, x, st = syn.gen_discrete_time_series(
        s=3,
        random=False,
        x=np.array([1.0, 2.0, 3.0]),
        start_time=datetime.datetime(2023, 1, 1),
    )
    npt.assert_array_equal(t, [0, 1, 2])
    npt.assert_array_equal(x, [1.0, 2.0, 3.0])
    assert st == datetime.datetime(2023, 1, 1)


def test_user_defined_missing_x_raises():
    with pytest.raises(ValueError):
        syn.gen_discrete_time_series(s=3, random=False)


def test_user_defined_wrong_size_x_raises():
    with pytest.raises(ValueError):
        syn.gen_discrete_time_series(s=3, random=False, x=np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_mismatched_t_size_raises():
    with pytest.raises(ValueError):
        syn.gen_discrete_time_series(s=5, random=True, t=np.arange(3))


def test_invalid_start_time_type_raises():
    with pytest.raises(ValueError):
        syn.gen_discrete_time_series(s=3, random=True, start_time="2023-01-01")


# ---------------------------------------------------------------------------
# File saving
# ---------------------------------------------------------------------------

def test_save_to_npz(tmp_path):
    fname = str(tmp_path / "series.npz")
    t, x, st = syn.gen_discrete_time_series(
        s=4,
        random=True,
        seed=5,
        start_time=datetime.datetime(2023, 6, 1),
        filename=fname,
    )
    loaded = np.load(fname, allow_pickle=True)
    npt.assert_array_equal(loaded["taxis"], t)
    npt.assert_allclose(loaded["data"], x, atol=1e-12)
