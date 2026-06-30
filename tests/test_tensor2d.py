"""Characterization tests for fiberis.analyzer.TensorProcessor.coreT2D.Tensor2D.

These pin the CURRENT observable behavior of the Tensor2D class (time- and
depth-varying 2x2 tensor fields) as a refactoring safety net.

Golden values were obtained by RUNNING the code. Small deterministic synthetic
tensors with analytically known projections/eigen-structure are used so the
characterization is strong. Where current behavior looks questionable it is
pinned anyway and annotated with ``# NOTE: possible bug``.
"""

import copy
import datetime

import numpy as np
import numpy.testing as npt
import pytest

from fiberis.analyzer.TensorProcessor.coreT2D import Tensor2D
from fiberis.analyzer.Data2D.core2D import Data2D


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def diag_tensor(sample_start_time):
    """A single (depth, time) diagonal tensor diag(2, 1).

    Known projection: n^T T n = 2 cos^2(theta) + 1 sin^2(theta) for a
    direction at angle theta from +x.
    """
    data = np.zeros((1, 1, 2, 2))
    data[0, 0] = np.array([[2.0, 0.0], [0.0, 1.0]])
    return Tensor2D(
        data=data,
        taxis=np.array([0.0]),
        daxis=np.array([5.0]),
        dim=2,
        start_time=sample_start_time,
        name="diag",
    )


@pytest.fixture
def field_tensor(sample_start_time):
    """A (2 depth, 4 time) field. Tensor at depth d is diag(1+d, 0)."""
    n_d, n_t = 2, 4
    data = np.zeros((n_d, n_t, 2, 2))
    for d in range(n_d):
        for t in range(n_t):
            data[d, t] = np.array([[1.0 + d, 0.2], [0.2, 0.5 * t]])
    return Tensor2D(
        data=data,
        taxis=np.array([0.0, 1.0, 2.0, 3.0]),
        daxis=np.array([10.0, 20.0]),
        dim=2,
        start_time=sample_start_time,
        name="field",
    )


# ---------------------------------------------------------------------------
# Construction / setters
# ---------------------------------------------------------------------------
class TestConstruction:
    def test_default_construction_is_empty(self):
        t = Tensor2D()
        assert t.data is None
        assert t.taxis is None
        assert t.daxis is None
        assert t.dim is None
        assert t.start_time is None
        assert t.name is None

    def test_set_data_infers_dim(self):
        t = Tensor2D().set_data(np.zeros((1, 1, 2, 2)))
        assert t.dim == 2

    def test_set_data_rejects_non_4d(self):
        with pytest.raises(TypeError):
            Tensor2D().set_data(np.zeros((2, 2, 2)))

    def test_set_data_rejects_non_square(self):
        with pytest.raises(ValueError):
            Tensor2D().set_data(np.zeros((1, 1, 2, 3)))

    def test_set_data_does_not_defensively_copy(self):
        # NOTE: possible bug -- Tensor2D.set_data stores the array by reference
        # (unlike CoreTensor.set_data which copies). External mutation leaks in.
        arr = np.zeros((1, 1, 2, 2))
        t = Tensor2D().set_data(arr)
        arr[0, 0, 0, 0] = 99.0
        assert t.data[0, 0, 0, 0] == 99.0

    def test_setters_return_self(self):
        t = Tensor2D()
        assert t.set_taxis(np.array([0.0])) is t
        assert t.set_daxis(np.array([1.0])) is t
        assert t.set_dim(2) is t
        assert t.set_name("x") is t
        st = datetime.datetime(2023, 1, 1)
        assert t.set_start_time(st) is t


# ---------------------------------------------------------------------------
# get_component
# ---------------------------------------------------------------------------
class TestGetComponent:
    def test_xx_component(self, field_tensor):
        c = field_tensor.get_component("xx")
        assert isinstance(c, Data2D)
        # depth 0 -> 1.0 across all times, depth 1 -> 2.0
        npt.assert_array_equal(c.data, np.array([[1.0] * 4, [2.0] * 4]))
        assert c.name == "field_xx"

    def test_offdiag_components(self, field_tensor):
        npt.assert_array_equal(
            field_tensor.get_component("xy").data, np.full((2, 4), 0.2)
        )
        npt.assert_array_equal(
            field_tensor.get_component("yx").data, np.full((2, 4), 0.2)
        )

    def test_yy_component_varies_with_time(self, field_tensor):
        # yy = 0.5 * t
        npt.assert_array_equal(
            field_tensor.get_component("yy").data,
            np.array([[0.0, 0.5, 1.0, 1.5]] * 2),
        )

    def test_tuple_component(self, field_tensor):
        npt.assert_array_equal(
            field_tensor.get_component((0, 1)).data, np.full((2, 4), 0.2)
        )

    def test_component_passes_axes_through(self, field_tensor):
        c = field_tensor.get_component("xx")
        npt.assert_array_equal(c.taxis, field_tensor.taxis)
        npt.assert_array_equal(c.daxis, field_tensor.daxis)
        assert c.start_time == field_tensor.start_time

    def test_invalid_component_string(self, field_tensor):
        with pytest.raises(ValueError):
            field_tensor.get_component("zz")

    def test_component_requires_data(self):
        with pytest.raises(ValueError):
            Tensor2D().get_component("xx")


# ---------------------------------------------------------------------------
# get_directional_component  (latest-commit feature)
# ---------------------------------------------------------------------------
class TestDirectionalComponent:
    def test_angle_0_from_x(self, diag_tensor):
        # direction +x -> T_xx = 2
        assert diag_tensor.get_directional_component(0).data[0, 0] == 2.0

    def test_angle_90_from_x(self, diag_tensor):
        # direction +y -> T_yy = 1
        npt.assert_allclose(
            diag_tensor.get_directional_component(90).data[0, 0], 1.0, atol=1e-12
        )

    def test_angle_45_from_x(self, diag_tensor):
        # 2 cos^2(45) + 1 sin^2(45) = 1.5
        npt.assert_allclose(
            diag_tensor.get_directional_component(45).data[0, 0], 1.5, atol=1e-12
        )

    def test_in_radians(self, diag_tensor):
        npt.assert_allclose(
            diag_tensor.get_directional_component(np.pi / 2, in_radians=True).data[0, 0],
            1.0,
            atol=1e-12,
        )

    def test_reference_axis_y_angle_0(self, diag_tensor):
        # angle 0 from +y -> direction +y -> 1.0
        npt.assert_allclose(
            diag_tensor.get_directional_component(0, reference_axis="y").data[0, 0],
            1.0,
            atol=1e-12,
        )

    def test_reference_axis_y_angle_90(self, diag_tensor):
        # angle 90 from +y -> direction +x -> 2.0
        npt.assert_allclose(
            diag_tensor.get_directional_component(90, reference_axis="y").data[0, 0],
            2.0,
            atol=1e-12,
        )

    def test_clockwise_from_x(self, diag_tensor):
        # clockwise 90 from +x -> direction (0,-1) -> T_yy = 1.0
        npt.assert_allclose(
            diag_tensor.get_directional_component(90, clockwise=True).data[0, 0],
            1.0,
            atol=1e-12,
        )

    def test_reference_axis_case_insensitive(self, diag_tensor):
        npt.assert_allclose(
            diag_tensor.get_directional_component(0, reference_axis="Y").data[0, 0],
            1.0,
            atol=1e-12,
        )

    def test_returns_data2d_with_axes(self, diag_tensor):
        dc = diag_tensor.get_directional_component(0)
        assert isinstance(dc, Data2D)
        npt.assert_array_equal(dc.taxis, diag_tensor.taxis)
        npt.assert_array_equal(dc.daxis, diag_tensor.daxis)
        assert dc.start_time == diag_tensor.start_time

    def test_default_output_name(self, diag_tensor):
        assert (
            diag_tensor.get_directional_component(0).name
            == "diag_directional_component"
        )

    def test_custom_output_name(self, diag_tensor):
        assert diag_tensor.get_directional_component(0, name="custom").name == "custom"

    def test_invalid_reference_axis(self, diag_tensor):
        with pytest.raises(ValueError):
            diag_tensor.get_directional_component(0, reference_axis="z")

    def test_requires_2d(self):
        t = Tensor2D(
            data=np.zeros((1, 1, 3, 3)),
            taxis=np.array([0.0]),
            daxis=np.array([0.0]),
            dim=3,
        )
        with pytest.raises(ValueError):
            t.get_directional_component(0)

    def test_requires_data(self):
        with pytest.raises(ValueError):
            Tensor2D().get_directional_component(0)

    def test_field_projection_matches_components(self, field_tensor):
        # Projecting at 0 deg from +x must equal the xx component everywhere.
        proj = field_tensor.get_directional_component(0).data
        npt.assert_allclose(proj, field_tensor.get_component("xx").data, atol=1e-12)


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------
class TestRotation:
    def test_rotate_returns_self(self, field_tensor):
        assert field_tensor.rotate_tensor(0) is field_tensor

    def test_rotate_requires_data(self):
        with pytest.raises(ValueError):
            Tensor2D(dim=2).rotate_tensor(10)

    def test_rotate_unsupported_dimension(self):
        t = Tensor2D(data=np.zeros((1, 1, 3, 3)), dim=3)
        with pytest.raises(ValueError):
            t.rotate_tensor(10)

    def test_rotate_90_uniaxial_current_behavior(self, sample_start_time):
        # NOTE: possible bug -- rotate_tensor uses
        #   np.einsum('ik,dtkj,lj->dtil', R, data, R.T)
        # which evaluates to R @ T @ R (NOT R @ T @ R^T as the docstring
        # claims). For T = [[1,0],[0,0]] rotated 90deg the active rotation
        # R T R^T would give [[0,0],[0,1]], but the current code returns
        # [[0,0],[0,-1]]. We pin the CURRENT (buggy) result.
        data = np.zeros((1, 1, 2, 2))
        data[0, 0] = np.array([[1.0, 0.0], [0.0, 0.0]])
        t = Tensor2D(
            data=data,
            taxis=np.array([0.0]),
            daxis=np.array([0.0]),
            dim=2,
            start_time=sample_start_time,
        )
        t.rotate_tensor(90)
        npt.assert_allclose(
            t.data[0, 0], np.array([[0.0, 0.0], [0.0, -1.0]]), atol=1e-12
        )

    def test_rotate_degrees_equals_radians(self, sample_start_time):
        data = np.zeros((1, 1, 2, 2))
        data[0, 0] = np.array([[1.0, 0.3], [0.3, -2.0]])
        t_deg = Tensor2D(
            data=data.copy(),
            taxis=np.array([0.0]),
            daxis=np.array([0.0]),
            dim=2,
        )
        t_rad = Tensor2D(
            data=data.copy(),
            taxis=np.array([0.0]),
            daxis=np.array([0.0]),
            dim=2,
        )
        t_deg.rotate_tensor(90)
        t_rad.rotate_tensor(np.pi / 2, in_radians=True)
        npt.assert_allclose(t_deg.data, t_rad.data, atol=1e-12)

    def test_rotate_360_is_identity(self, sample_start_time):
        data = np.zeros((1, 1, 2, 2))
        data[0, 0] = np.array([[1.0, 0.3], [0.3, -2.0]])
        t = Tensor2D(
            data=data.copy(),
            taxis=np.array([0.0]),
            daxis=np.array([0.0]),
            dim=2,
        )
        original = t.data.copy()
        t.rotate_tensor(360)
        npt.assert_allclose(t.data, original, atol=1e-12)


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------
class TestSelectTime:
    def test_select_time_floats(self, field_tensor):
        t = copy.deepcopy(field_tensor)
        t.select_time(1.0, 2.0)
        npt.assert_array_equal(t.taxis, np.array([1.0, 2.0]))
        assert t.data.shape == (2, 2, 2, 2)

    def test_select_time_datetimes(self, field_tensor, sample_start_time):
        t = copy.deepcopy(field_tensor)
        t.select_time(
            sample_start_time + datetime.timedelta(seconds=1),
            sample_start_time + datetime.timedelta(seconds=2),
        )
        npt.assert_array_equal(t.taxis, np.array([1.0, 2.0]))

    def test_select_time_inclusive_bounds(self, field_tensor):
        t = copy.deepcopy(field_tensor)
        t.select_time(0.0, 3.0)
        npt.assert_array_equal(t.taxis, field_tensor.taxis)

    def test_select_time_returns_self(self, field_tensor):
        t = copy.deepcopy(field_tensor)
        assert t.select_time(0.0, 1.0) is t

    def test_select_time_requires_start_time(self):
        t = Tensor2D(
            data=np.zeros((1, 2, 2, 2)),
            taxis=np.array([0.0, 1.0]),
            daxis=np.array([0.0]),
            dim=2,
        )
        with pytest.raises(ValueError):
            t.select_time(0, 1)


class TestSelectDepth:
    def test_select_depth(self, field_tensor):
        t = copy.deepcopy(field_tensor)
        t.select_depth(15.0, 25.0)
        npt.assert_array_equal(t.daxis, np.array([20.0]))
        assert t.data.shape == (1, 4, 2, 2)

    def test_select_depth_inclusive(self, field_tensor):
        t = copy.deepcopy(field_tensor)
        t.select_depth(10.0, 20.0)
        npt.assert_array_equal(t.daxis, field_tensor.daxis)

    def test_select_depth_returns_self(self, field_tensor):
        t = copy.deepcopy(field_tensor)
        assert t.select_depth(0.0, 100.0) is t

    def test_select_depth_requires_data(self):
        with pytest.raises(ValueError):
            Tensor2D(daxis=np.array([1.0])).select_depth(0, 1)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
class TestIO:
    def test_save_load_roundtrip(self, field_tensor, tmp_path):
        fname = str(tmp_path / "t2d.npz")
        assert field_tensor.savez(fname) is None
        loaded = Tensor2D().load_npz(fname)
        npt.assert_array_equal(loaded.data, field_tensor.data)
        npt.assert_array_equal(loaded.taxis, field_tensor.taxis)
        npt.assert_array_equal(loaded.daxis, field_tensor.daxis)
        assert loaded.dim == 2
        assert loaded.name == "field"
        assert loaded.start_time == field_tensor.start_time

    def test_savez_appends_extension(self, field_tensor, tmp_path):
        fname = str(tmp_path / "noext")
        field_tensor.savez(fname)
        assert (tmp_path / "noext.npz").exists()

    def test_savez_requires_essentials(self, tmp_path):
        with pytest.raises(ValueError):
            Tensor2D().savez(str(tmp_path / "empty.npz"))

    def test_load_appends_extension(self, field_tensor, tmp_path):
        field_tensor.savez(str(tmp_path / "x.npz"))
        loaded = Tensor2D().load_npz(str(tmp_path / "x"))
        assert loaded.dim == 2


# ---------------------------------------------------------------------------
# __str__
# ---------------------------------------------------------------------------
class TestStr:
    def test_str_populated(self, field_tensor):
        s = str(field_tensor)
        assert "Tensor2D Object: field" in s
        assert "Data Shape: (2, 4, 2, 2)" in s
        assert "Time Axis Size: 4" in s
        assert "Depth Axis Size: 2" in s
        assert "Tensor Dimension: 2" in s

    def test_str_empty(self):
        s = str(Tensor2D())
        assert "Tensor2D Object: Unnamed" in s
        assert "Data Shape: N/A" in s
        assert "Start Time: N/A" in s
