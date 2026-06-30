"""
Characterization (golden-master) tests for the lightweight 1D simulator core:
    - fiberis.simulator.core.pds   (PDS1D_SingleSource / PDS1D_MultiSource)
    - fiberis.simulator.core.bcs   (BoundaryCondition)

These tests pin the CURRENT behavior of the code so a pure refactor can be
validated against them. Golden values were obtained by RUNNING the code on
small deterministic problems, not by hand-derivation. Where the current
behavior looks suspicious it is pinned anyway with a `# NOTE: possible bug`.

Run:
    cd /home/user/fibeRIS && python3 -m pytest tests/test_simulator_core.py -q
"""

import datetime

import numpy as np
import numpy.testing as npt
import pytest

from fiberis.simulator.core import pds as pdsmod
from fiberis.simulator.core.bcs import BoundaryCondition
from fiberis.analyzer.Data1D.core1D import Data1D


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def make_source(data=(0.0, 10.0, 10.0), taxis=(0.0, 5.0, 10.0)):
    """A small real Data1D source object (provides get_value_by_time / taxis)."""
    return Data1D(
        data=np.asarray(data, dtype=float),
        taxis=np.asarray(taxis, dtype=float),
        start_time=datetime.datetime(2020, 1, 1),
    )


def make_single(lbc="Dirichlet", rbc="Dirichlet", nx=5, diff=1.0, sourceidx=2):
    p = pdsmod.PDS1D_SingleSource()
    p.set_mesh(np.linspace(0.0, float(nx - 1), nx))
    p.set_source(make_source())
    p.set_bcs(lbc, rbc)
    p.set_initial(np.zeros(nx))
    p.set_diffusivity(diff)
    p.set_sourceidx(sourceidx)
    p.set_t0(0.0)
    return p


# --------------------------------------------------------------------------
# BoundaryCondition (bcs.py)
# --------------------------------------------------------------------------
class TestBoundaryCondition:
    def test_dirichlet_apply(self):
        A = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])
        b = np.array([1.0, 2.0, 3.0])
        bc = BoundaryCondition(type="Dirichlet", value=5.0, idx=0)
        bc.set_matrix(A, b)
        bc.apply()
        npt.assert_array_equal(A[0], np.array([1.0, 0.0, 0.0]))
        npt.assert_array_equal(b, np.array([5.0, 2.0, 3.0]))

    def test_neumann_apply_left(self):
        A = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])
        b = np.array([1.0, 2.0, 3.0])
        bc = BoundaryCondition(type="Neumann", value=0.0, idx=0)
        bc.set_matrix(A, b)
        bc.apply()
        # NOTE: BoundaryCondition.apply() Neumann uses diag=+1, neighbour=-1.
        # This is the OPPOSITE sign convention from the inline Neumann in
        # matbuilder (diag=-1, neighbour=+1). Pinned as-is.
        npt.assert_array_equal(A[0], np.array([1.0, -1.0, 0.0]))
        npt.assert_array_equal(b, np.array([0.0, 2.0, 3.0]))

    def test_neumann_apply_right(self):
        A = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])
        b = np.array([1.0, 2.0, 3.0])
        bc = BoundaryCondition(type="Neumann", value=0.0, idx=2)
        bc.set_matrix(A, b)
        bc.apply()
        npt.assert_array_equal(A[2], np.array([0.0, -1.0, 1.0]))
        npt.assert_array_equal(b, np.array([1.0, 2.0, 0.0]))

    def test_set_matrix_nonsquare_raises(self):
        bc = BoundaryCondition()
        with pytest.raises(ValueError, match="not square"):
            bc.set_matrix(np.zeros((2, 3)), np.zeros(2))

    def test_set_matrix_vecB_not_vector_raises(self):
        bc = BoundaryCondition()
        with pytest.raises(ValueError, match="not a vector"):
            bc.set_matrix(np.eye(2), np.zeros((2, 2)))

    def test_set_matrix_size_mismatch_raises(self):
        bc = BoundaryCondition()
        with pytest.raises(ValueError, match="does not match"):
            bc.set_matrix(np.eye(3), np.zeros(2))

    def test_unknown_bc_type_raises(self):
        bc = BoundaryCondition(type="foo", idx=0)
        bc.set_matrix(np.eye(2), np.zeros(2))
        with pytest.raises(ValueError, match="Unknown BC type"):
            bc.apply()

    def test_pml_type_is_noop(self):
        A = np.eye(3)
        b = np.zeros(3)
        bc = BoundaryCondition(type="PML", idx=0)
        bc.set_matrix(A, b)
        # PML branch is `pass`; nothing changes and no error.
        assert bc.apply() is None
        npt.assert_array_equal(A, np.eye(3))


# --------------------------------------------------------------------------
# PDS1D_SingleSource: configuration / self-check (pds.py)
# --------------------------------------------------------------------------
class TestPDSSingleConfig:
    def test_diffusivity_scalar_broadcast(self):
        p = pdsmod.PDS1D_SingleSource()
        p.set_mesh(np.linspace(0.0, 4.0, 5))
        p.set_diffusivity(2.0)
        npt.assert_array_equal(p.diffusivity, np.full(5, 2.0))

    def test_diffusivity_array_ok(self):
        p = pdsmod.PDS1D_SingleSource()
        p.set_mesh(np.linspace(0.0, 4.0, 5))
        p.set_diffusivity([1.0, 2.0, 3.0, 4.0, 5.0])
        npt.assert_array_equal(p.diffusivity, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

    def test_diffusivity_wrong_length_raises(self):
        p = pdsmod.PDS1D_SingleSource()
        p.set_mesh(np.linspace(0.0, 4.0, 5))
        with pytest.raises(ValueError, match="single scalar value or an array"):
            p.set_diffusivity([1.0, 2.0])

    def test_self_check_all_pass(self):
        p = make_single()
        assert p.self_check() is True

    def test_self_check_unset_fails(self):
        p = pdsmod.PDS1D_SingleSource()
        assert p.self_check() is False

    def test_self_check_unknown_param(self):
        p = make_single()
        assert p.self_check("does_not_exist") is False

    def test_check_mesh_too_short(self):
        p = pdsmod.PDS1D_SingleSource()
        p.set_mesh(np.array([1.0]))
        ok, msg = p._check_mesh()
        assert ok is False and "at least 2" in msg

    def test_check_bc_invalid(self):
        p = pdsmod.PDS1D_SingleSource()
        p.set_bcs("Dirichlet", "Bogus")
        ok, msg = p._check_bc()
        assert ok is False and "Invalid right BC" in msg

    def test_solve_returns_none_when_self_check_fails(self):
        p = pdsmod.PDS1D_SingleSource()
        # Nothing configured -> self_check fails -> solve() returns None early.
        assert p.solve(dt=0.1, t_total=1.0) is None

    def test_reset_clears_state(self):
        p = make_single()
        p.snapshot = "x"
        p.taxis = "y"
        p.history = ["a", "b"]
        p.reset()
        assert p.snapshot is None
        assert p.taxis is None
        assert p.history == []


# --------------------------------------------------------------------------
# PDS1D_SingleSource: full solve (golden master)
# --------------------------------------------------------------------------
class TestPDSSingleSolve:
    def test_solve_shapes_and_taxis(self):
        p = make_single()
        p.solve(dt=0.5, t_total=2.0, mode="implicit")
        # snapshot list starts with initial, then one append per accepted step.
        assert p.snapshot.shape == (5, 5)
        npt.assert_array_almost_equal(p.taxis, np.array([0.0, 0.5, 1.0, 1.5, 2.0]))

    def test_solve_golden_values_dirichlet(self):
        p = make_single()
        p.solve(dt=0.5, t_total=2.0, mode="implicit")
        npt.assert_allclose(
            p.snapshot[-1],
            np.array([0.0, 1.0625, 3.0, 1.0625, 0.0]),
            rtol=0, atol=1e-9,
        )

    def test_solve_source_column_matches_source_values(self):
        # Source idx is pinned to the interpolated source value at each step.
        p = make_single()
        p.solve(dt=0.5, t_total=2.0, mode="implicit")
        npt.assert_allclose(
            p.get_val_at_source_idx(),
            np.array([0.0, 0.0, 1.0, 2.0, 3.0]),
            rtol=0, atol=1e-9,
        )

    def test_solution_symmetry_dirichlet(self):
        # Symmetric setup -> solution symmetric about the source index.
        p = make_single()
        p.solve(dt=0.5, t_total=2.0, mode="implicit")
        final = p.snapshot[-1]
        npt.assert_allclose(final, final[::-1], rtol=0, atol=1e-12)

    def test_default_mode_is_implicit(self):
        # Omitting `mode` defaults to the implicit solver; results match.
        p_default = make_single()
        p_default.solve(dt=0.5, t_total=2.0)
        p_imp = make_single()
        p_imp.solve(dt=0.5, t_total=2.0, mode="implicit")
        npt.assert_allclose(p_default.snapshot, p_imp.snapshot, rtol=0, atol=1e-12)

    def test_get_val_at_time(self):
        p = make_single()
        p.solve(dt=0.5, t_total=2.0, mode="implicit")
        # Closest snapshot to t=1.0 is index 2.
        npt.assert_allclose(p.get_val_at_time(1.0), p.snapshot[2], rtol=0, atol=1e-12)

    def test_get_solution_tuple(self):
        p = make_single()
        p.solve(dt=0.5, t_total=2.0, mode="implicit")
        snap, taxis = p.get_solution()
        assert snap is p.snapshot and taxis is p.taxis


# --------------------------------------------------------------------------
# PDS1D_MultiSource (golden master)
# --------------------------------------------------------------------------
class TestPDSMultiSource:
    def make_multi(self):
        s1 = make_source(data=(0.0, 5.0, 5.0))
        s2 = make_source(data=(0.0, -3.0, -3.0))
        p = pdsmod.PDS1D_MultiSource()
        p.set_mesh(np.linspace(0.0, 6.0, 7))
        p.set_source([s1, s2])
        p.set_bcs("Dirichlet", "Dirichlet")
        p.set_initial(np.zeros(7))
        p.set_diffusivity(1.0)
        p.set_sourceidx([2, 4])
        p.set_t0(0.0)
        return p

    def test_multi_solve_shape(self):
        p = self.make_multi()
        p.solve(dt=0.5, t_total=1.5, mode="implicit")
        assert p.snapshot.shape == (4, 7)
        npt.assert_array_almost_equal(p.taxis, np.array([0.0, 0.5, 1.0, 1.5]))

    def test_multi_solve_golden_final(self):
        p = self.make_multi()
        p.solve(dt=0.5, t_total=1.5, mode="implicit")
        npt.assert_allclose(
            p.snapshot[-1],
            np.array([0.0, 0.3125, 1.0, 0.125, -0.6, -0.1875, 0.0]),
            rtol=0, atol=1e-9,
        )

    def test_multi_source_columns_pinned(self):
        p = self.make_multi()
        p.solve(dt=0.5, t_total=1.5, mode="implicit")
        npt.assert_allclose(
            p.get_val_at_idx(2), np.array([0.0, 0.0, 0.5, 1.0]), rtol=0, atol=1e-9
        )
        npt.assert_allclose(
            p.get_val_at_idx(4), np.array([0.0, 0.0, -0.3, -0.6]), rtol=0, atol=1e-9
        )

    def test_multi_sourceidx_is_ndarray(self):
        p = self.make_multi()
        # PDS1D_MultiSource.set_sourceidx converts to a numpy array.
        assert isinstance(p.sourceidx, np.ndarray)
        npt.assert_array_equal(p.sourceidx, np.array([2, 4]))

    def test_multi_check_source_requires_list(self):
        p = pdsmod.PDS1D_MultiSource()
        p.set_source(make_source())  # a single Data1D, not a list
        ok, msg = p._check_source()
        assert ok is False and "must be a list" in msg
