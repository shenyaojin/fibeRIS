"""
Characterization (golden-master) tests for the time-sampling optimizer:
    - fiberis.simulator.optimizer.tso  (adjust_dt, time_sampling_optimizer)

These exercise the real numerics; tso has no external dependencies, so a
lightweight fake pds object (only the attributes/methods tso touches) is
sufficient. Golden values obtained by RUNNING the code.

Run:
    cd /home/user/fibeRIS && python3 -m pytest tests/test_simulator_optimizer.py -q
"""

import numpy as np
import numpy.testing as npt
import pytest

from fiberis.simulator.optimizer import tso


def test_import_smoke():
    assert hasattr(tso, "time_sampling_optimizer")
    assert hasattr(tso, "adjust_dt")


# --------------------------------------------------------------------------
# adjust_dt
# --------------------------------------------------------------------------
class TestAdjustDt:
    def test_error_equals_tol_keeps_safety_factor(self):
        # error == tol -> ratio == 1 -> dt * safety_factor (0.9).
        assert tso.adjust_dt(1.0, 1e-3) == pytest.approx(0.9)

    def test_error_larger_than_tol_shrinks(self):
        # Pinned golden: dt=1, error=1e-2, p=2, tol=1e-3, sf=0.9.
        assert tso.adjust_dt(1.0, 1e-2) == pytest.approx(0.28460498941515416, rel=1e-12)

    def test_tiny_error_treated_as_ratio_one(self):
        # error < 1e-14 -> ratio forced to 1.0 -> 0.9.
        assert tso.adjust_dt(1.0, 1e-16) == pytest.approx(0.9)

    def test_clamp_to_max_dt(self):
        # Large growth is clamped to max_dt (default 40).
        assert tso.adjust_dt(100.0, 1e-9) == pytest.approx(40.0)

    def test_clamp_to_min_dt(self):
        # Large error forces shrink, clamped to min_dt (default 1e-4).
        assert tso.adjust_dt(1e-6, 1.0) == pytest.approx(1e-4)

    def test_custom_kwargs_override(self):
        # safety_factor and max_dt overridable via kwargs.
        out = tso.adjust_dt(10.0, 1e-9, safety_factor=0.5, max_dt=100.0)
        # ratio capped by tol/error huge -> dt*0.5*ratio clamped to max_dt=100.
        assert out == pytest.approx(100.0)


# --------------------------------------------------------------------------
# time_sampling_optimizer
# --------------------------------------------------------------------------
class FakePDS:
    """Minimal stand-in exposing only what time_sampling_optimizer uses."""

    def __init__(self):
        self.snapshot = [np.zeros(3)]
        self.taxis = [0.0]

    def record_log(self, *args):
        pass


class TestTimeSamplingOptimizer:
    def test_accept_when_error_below_tol(self):
        fp = FakePDS()
        full = np.array([1.0, 2.0, 3.0])
        half = np.array([1.0, 2.0, 3.0])  # error == 0 -> accept
        next_dt = tso.time_sampling_optimizer(fp, full, half, 0.5)
        # Accepted: snapshot/taxis appended.
        assert len(fp.snapshot) == 2
        npt.assert_array_equal(fp.snapshot[-1], full)
        assert fp.taxis == [0.0, 0.5]
        # error < 1e-14 -> ratio 1 -> next dt = 0.5 * 0.9 = 0.45.
        assert next_dt == pytest.approx(0.45)

    def test_reject_when_error_above_tol(self):
        fp = FakePDS()
        full = np.array([1.0, 2.0, 3.0])
        half = np.array([2.0, 4.0, 6.0])  # error == 1.0 -> reject
        next_dt = tso.time_sampling_optimizer(fp, full, half, 0.5)
        # Rejected: snapshot/taxis unchanged.
        assert len(fp.snapshot) == 1
        assert fp.taxis == [0.0]
        # Golden next dt for error=1.0 from dt=0.5.
        assert next_dt == pytest.approx(0.014230249470757706, rel=1e-12)

    def test_error_at_exactly_tol_is_accepted(self):
        # tol default 1e-3; construct vectors with relative error exactly 1e-3.
        fp = FakePDS()
        full = np.array([1000.0, 0.0, 0.0])
        half = np.array([999.0, 0.0, 0.0])  # ||diff||/||full|| = 1/1000 = 1e-3
        err = np.linalg.norm(full - half) / np.linalg.norm(full)
        assert err == pytest.approx(1e-3)
        tso.time_sampling_optimizer(fp, full, half, 0.5)
        # error <= tol -> accepted.
        assert len(fp.snapshot) == 2
