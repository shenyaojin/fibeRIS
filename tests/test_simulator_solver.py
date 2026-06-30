"""
Characterization (golden-master) tests for the 1D simulator solver layer:
    - fiberis.simulator.solver.matbuilder      (matrix assembly + PML)
    - fiberis.simulator.solver.PDESolver_IMP   (implicit / linear solve)
    - fiberis.simulator.solver.PDESolver_EXP   (explicit Jacobi iteration)

Golden values obtained by RUNNING the code on small deterministic problems.

Run:
    cd /home/user/fibeRIS && python3 -m pytest tests/test_simulator_solver.py -q
"""

import datetime

import numpy as np
import numpy.testing as npt
import pytest

from fiberis.simulator.core import pds as pdsmod
from fiberis.simulator.solver import matbuilder
from fiberis.simulator.solver.PDESolver_IMP import solver_implicit
from fiberis.simulator.solver.PDESolver_EXP import solver_explicit
from fiberis.analyzer.Data1D.core1D import Data1D


def make_source(data=(0.0, 10.0, 10.0), taxis=(0.0, 5.0, 10.0)):
    return Data1D(
        data=np.asarray(data, dtype=float),
        taxis=np.asarray(taxis, dtype=float),
        start_time=datetime.datetime(2020, 1, 1),
    )


def make_single(lbc="Neumann", rbc="Neumann", nx=5, diff=1.0, sourceidx=2):
    """Configured PDS object primed for a single matrix-builder call (1 step)."""
    p = pdsmod.PDS1D_SingleSource()
    p.set_mesh(np.linspace(0.0, float(nx - 1), nx))
    p.set_source(make_source())
    p.set_bcs(lbc, rbc)
    p.set_initial(np.zeros(nx))
    p.set_diffusivity(diff)
    p.set_sourceidx(sourceidx)
    p.set_t0(0.0)
    # Prime the transient state matbuilder reads (normally set up in solve()).
    p.taxis = [0.0]
    p.snapshot = [np.zeros(nx)]
    return p


# --------------------------------------------------------------------------
# PDESolver_IMP.solver_implicit
# --------------------------------------------------------------------------
class TestImplicitSolver:
    A = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])
    b = np.array([1.0, 2.0, 3.0])
    expected = np.array([2.5, 4.0, 3.5])

    def test_numpy_solver(self):
        x = solver_implicit(self.A.copy(), self.b.copy(), solver="numpy")
        npt.assert_allclose(x, self.expected, rtol=0, atol=1e-12)

    def test_direct_solver(self):
        x = solver_implicit(self.A.copy(), self.b.copy(), solver="direct")
        npt.assert_allclose(x, self.expected, rtol=0, atol=1e-12)

    def test_scipy_solver(self):
        x = solver_implicit(self.A.copy(), self.b.copy(), solver="scipy")
        npt.assert_allclose(x, self.expected, rtol=0, atol=1e-12)

    def test_default_solver_is_scipy(self):
        # Default kwarg solver='scipy'; sparse spsolve gives the same answer.
        x = solver_implicit(self.A.copy(), self.b.copy())
        npt.assert_allclose(x, self.expected, rtol=0, atol=1e-12)

    def test_invalid_solver_raises(self):
        with pytest.raises(ValueError, match="Invalid solver type"):
            solver_implicit(self.A.copy(), self.b.copy(), solver="bogus")


# --------------------------------------------------------------------------
# PDESolver_EXP.solver_explicit (Jacobi)
# --------------------------------------------------------------------------
class TestExplicitSolver:
    def test_jacobi_diagonally_dominant(self):
        A = np.array([[4.0, 1.0, 0.0], [1.0, 5.0, 1.0], [0.0, 1.0, 3.0]])
        b = np.array([5.0, 7.0, 4.0])
        x = solver_explicit(A, b)
        # Jacobi should converge to the true linear-system solution.
        npt.assert_allclose(x, np.linalg.solve(A, b), rtol=0, atol=1e-8)

    def test_identity_matrix(self):
        A = np.eye(4)
        b = np.array([1.0, -2.0, 3.0, 4.0])
        x = solver_explicit(A, b)
        npt.assert_allclose(x, b, rtol=0, atol=1e-12)

    def test_non_convergent_returns_after_max_iter(self):
        # Strongly non-diagonally-dominant -> Jacobi diverges; the function does
        # NOT raise, it returns the last iterate after max_iter.
        # NOTE: possible bug -- silent return on non-convergence.
        A = np.array([[1.0, 5.0], [5.0, 1.0]])
        b = np.array([1.0, 1.0])
        x = solver_explicit(A, b, max_iter=5)
        assert x.shape == (2,)
        assert np.all(np.isfinite(x))


# --------------------------------------------------------------------------
# matbuilder.matrix_builder_1d_single_source
# --------------------------------------------------------------------------
class TestMatBuilderSingle:
    def test_assembly_neumann_golden(self):
        p = make_single(lbc="Neumann", rbc="Neumann", nx=5, diff=1.0, sourceidx=2)
        A, b = matbuilder.matrix_builder_1d_single_source(p, 0.5)
        expected_A = np.array(
            [
                [-1.0, 1.0, 0.0, 0.0, 0.0],
                [-0.5, 2.0, -0.5, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -0.5, 2.0, -0.5],
                [0.0, 0.0, 0.0, 1.0, -1.0],
            ]
        )
        npt.assert_allclose(A, expected_A, rtol=0, atol=1e-12)
        # b is the previous snapshot (all zeros) with source value at sourceidx.
        # At t=0 the source interpolates to 0.0.
        npt.assert_allclose(b, np.zeros(5), rtol=0, atol=1e-12)

    def test_assembly_shapes(self):
        p = make_single(nx=6)
        A, b = matbuilder.matrix_builder_1d_single_source(p, 0.3)
        assert A.shape == (6, 6)
        assert b.shape == (6,)

    def test_source_row_is_identity(self):
        # The source row is overwritten: zeroed then diagonal set to 1.
        p = make_single(sourceidx=2)
        A, _ = matbuilder.matrix_builder_1d_single_source(p, 0.5)
        expected_row = np.zeros(5)
        expected_row[2] = 1.0
        npt.assert_array_equal(A[2], expected_row)

    def test_source_value_in_b(self):
        # Prime taxis so the source interpolates to a nonzero value.
        p = make_single(sourceidx=2)
        p.taxis = [5.0]  # source = 10.0 at t=5
        A, b = matbuilder.matrix_builder_1d_single_source(p, 0.5)
        assert b[2] == pytest.approx(10.0)

    def test_dirichlet_boundary_rows(self):
        p = make_single(lbc="Dirichlet", rbc="Dirichlet", nx=5, sourceidx=2)
        A, b = matbuilder.matrix_builder_1d_single_source(p, 0.5)
        assert A[0, 0] == pytest.approx(1.0)
        assert A[-1, -1] == pytest.approx(1.0)
        assert b[0] == pytest.approx(0.0)
        assert b[-1] == pytest.approx(0.0)

    def test_pml_adds_to_diagonal(self):
        p = make_single(sourceidx=2)
        A0, _ = matbuilder.matrix_builder_1d_single_source(p, 0.5, pml_thickness=0.0)
        # Re-prime (matbuilder mutates the source row but reads snapshot copy).
        p2 = make_single(sourceidx=2)
        Apml, _ = matbuilder.matrix_builder_1d_single_source(
            p2, 0.5, pml_thickness=2.0, sigma_max=3.0
        )
        # PML increases diagonal entries near the boundaries.
        assert Apml[0, 0] >= A0[0, 0]


# --------------------------------------------------------------------------
# matbuilder.matrix_builder_1d_multi_source
# --------------------------------------------------------------------------
class TestMatBuilderMulti:
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
        p.taxis = [5.0]  # so sources interpolate to 5.0 and -3.0
        p.snapshot = [np.zeros(7)]
        return p

    def test_multi_source_rows_and_values(self):
        p = self.make_multi()
        A, b = matbuilder.matrix_builder_1d_multi_source(p, 0.5)
        assert A.shape == (7, 7) and b.shape == (7,)
        # Each source row is identity at its index.
        for idx in (2, 4):
            row = np.zeros(7)
            row[idx] = 1.0
            npt.assert_array_equal(A[idx], row)
        assert b[2] == pytest.approx(5.0)
        assert b[4] == pytest.approx(-3.0)


# --------------------------------------------------------------------------
# matbuilder.build_pml_sigma
# --------------------------------------------------------------------------
class TestBuildPMLSigma:
    def test_zero_thickness_all_zero(self):
        mesh = np.linspace(0.0, 10.0, 11)
        sigma = matbuilder.build_pml_sigma(mesh, 0.0, 2.0)
        npt.assert_array_equal(sigma, np.zeros(11))

    def test_symmetric_ramp(self):
        mesh = np.linspace(0.0, 10.0, 11)
        sigma = matbuilder.build_pml_sigma(mesh, 3.0, 2.0)
        expected = np.array(
            [2.0, 2.0 / 3.0 * 2, 2.0 / 3.0, 0, 0, 0, 0, 0, 2.0 / 3.0, 4.0 / 3.0, 2.0]
        )
        npt.assert_allclose(sigma, expected, rtol=0, atol=1e-12)
        # Boundaries get max sigma; interior is zero.
        assert sigma[0] == pytest.approx(2.0)
        assert sigma[-1] == pytest.approx(2.0)
        assert sigma[5] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Cross-check: implicit vs explicit through a full solve
# --------------------------------------------------------------------------
class TestImplicitVsExplicit:
    def _solve(self, mode):
        s = make_source()
        p = pdsmod.PDS1D_SingleSource()
        p.set_mesh(np.linspace(0.0, 4.0, 5))
        p.set_source(s)
        p.set_bcs("Dirichlet", "Dirichlet")
        p.set_initial(np.zeros(5))
        p.set_diffusivity(1.0)
        p.set_sourceidx(2)
        p.set_t0(0.0)
        p.solve(dt=0.2, t_total=1.0, mode=mode)
        return p

    def test_implicit_and_explicit_agree(self):
        pi = self._solve("implicit")
        pe = self._solve("explicit")
        assert pi.snapshot.shape == pe.snapshot.shape
        # Jacobi (explicit) solves the same linear system to tight tolerance.
        npt.assert_allclose(pi.snapshot, pe.snapshot, rtol=0, atol=1e-6)

    def test_implicit_golden_final(self):
        pi = self._solve("implicit")
        npt.assert_allclose(
            pi.snapshot[-1],
            np.array([0.0, 0.4301541, 1.6, 0.4301541, 0.0]),
            rtol=0, atol=1e-6,
        )
