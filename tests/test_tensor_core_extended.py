"""Characterization tests for fiberis.analyzer.TensorProcessor.coreT.CoreTensor.

These pin the CURRENT observable behavior of the base CoreTensor class as a
safety net before refactoring. They intentionally complement (and do not
duplicate) the existing tests in tests/test_tensor.py.

Golden values were obtained by RUNNING the code, not by guessing. Where the
current behavior looks questionable it is pinned anyway and annotated with a
``# NOTE: possible bug`` comment.
"""

import datetime

import numpy as np
import numpy.testing as npt
import pytest

from fiberis.analyzer.TensorProcessor.coreT import CoreTensor


# ---------------------------------------------------------------------------
# Construction / defaults
# ---------------------------------------------------------------------------
class TestConstruction:
    def test_default_construction_is_empty(self):
        ct = CoreTensor()
        assert ct.data is None
        assert ct.taxis is None
        assert ct.dim is None
        assert ct.start_time is None
        assert ct.name == "Default Tensor Data"

    def test_named_construction(self):
        ct = CoreTensor(name="my_tensor")
        assert ct.name == "my_tensor"

    def test_history_records_on_init(self):
        ct = CoreTensor()
        # An InfoManagementSystem record is created on init.
        assert ct.history is not None


# ---------------------------------------------------------------------------
# Setters: validation and method chaining
# ---------------------------------------------------------------------------
class TestSetters:
    def test_set_data_infers_dim_when_unset(self):
        ct = CoreTensor()
        ret = ct.set_data(np.zeros((3, 3, 4)))
        assert ret is ct  # method chaining
        assert ct.dim == 3
        assert ct.data.shape == (3, 3, 4)

    def test_set_data_defensive_copy(self):
        arr = np.zeros((2, 2, 2))
        ct = CoreTensor().set_data(arr)
        arr[0, 0, 0] = 99.0
        # CoreTensor.set_data copies, so external mutation does not leak in.
        assert ct.data[0, 0, 0] == 0.0

    def test_set_data_rejects_non_ndarray(self):
        with pytest.raises(TypeError):
            CoreTensor().set_data([[1, 2], [3, 4]])

    def test_set_data_rejects_non_3d(self):
        with pytest.raises(ValueError):
            CoreTensor().set_data(np.zeros((2, 2)))

    def test_set_data_rejects_non_square(self):
        with pytest.raises(ValueError):
            CoreTensor().set_data(np.zeros((2, 3, 4)))

    def test_set_data_dim_mismatch(self):
        ct = CoreTensor(dim=2)
        with pytest.raises(ValueError):
            ct.set_data(np.zeros((3, 3, 4)))

    def test_set_data_taxis_length_mismatch(self):
        ct = CoreTensor()
        ct.set_taxis(np.array([0.0, 1.0, 2.0]))
        with pytest.raises(ValueError):
            ct.set_data(np.zeros((2, 2, 5)))

    def test_set_taxis_defensive_copy_and_chaining(self):
        arr = np.array([0.0, 1.0, 2.0])
        ct = CoreTensor()
        ret = ct.set_taxis(arr)
        assert ret is ct
        arr[0] = 99.0
        assert ct.taxis[0] == 0.0

    def test_set_taxis_rejects_non_ndarray(self):
        with pytest.raises(TypeError):
            CoreTensor().set_taxis([0, 1, 2])

    def test_set_taxis_rejects_non_1d(self):
        with pytest.raises(ValueError):
            CoreTensor().set_taxis(np.zeros((2, 2)))

    def test_set_taxis_rejects_decreasing(self):
        with pytest.raises(ValueError):
            CoreTensor().set_taxis(np.array([0.0, 2.0, 1.0]))

    def test_set_taxis_allows_repeated_values(self):
        # diff == 0 is NOT < 0, so equal consecutive values are accepted.
        ct = CoreTensor().set_taxis(np.array([0.0, 1.0, 1.0, 2.0]))
        npt.assert_array_equal(ct.taxis, np.array([0.0, 1.0, 1.0, 2.0]))

    def test_set_taxis_length_mismatch_with_data(self):
        ct = CoreTensor().set_data(np.zeros((2, 2, 3)))
        with pytest.raises(ValueError):
            ct.set_taxis(np.array([0.0, 1.0]))

    def test_set_dim_rejects_non_python_int(self):
        # numpy integer is NOT a python int -> TypeError under current behavior.
        with pytest.raises(TypeError):
            CoreTensor().set_dim(np.int64(2))  # NOTE: possible bug (np.int rejected)

    def test_set_dim_accepts_python_int(self):
        ct = CoreTensor().set_dim(3)
        assert ct.dim == 3

    def test_set_dim_mismatch_with_data(self):
        ct = CoreTensor().set_data(np.zeros((2, 2, 2)))
        with pytest.raises(ValueError):
            ct.set_dim(3)

    def test_set_start_time_validation(self):
        with pytest.raises(TypeError):
            CoreTensor().set_start_time("2023-01-01")
        st = datetime.datetime(2023, 1, 1)
        ct = CoreTensor().set_start_time(st)
        assert ct.start_time == st

    def test_set_name_validation(self):
        with pytest.raises(TypeError):
            CoreTensor().set_name(123)
        ct = CoreTensor().set_name("foo")
        assert ct.name == "foo"


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------
class TestRotation:
    def test_rotate_returns_none(self):
        ct = CoreTensor(data=np.zeros((2, 2, 1)), taxis=np.array([0.0]), dim=2)
        assert ct.rotate_tensor(0) is None

    def test_rotate_accepts_int_degree(self):
        ct = CoreTensor(data=np.zeros((2, 2, 1)), taxis=np.array([0.0]), dim=2)
        ct.rotate_tensor(0)  # int accepted, no error

    def test_rotate_rejects_non_numeric(self):
        ct = CoreTensor(data=np.zeros((2, 2, 1)), taxis=np.array([0.0]), dim=2)
        with pytest.raises(TypeError):
            ct.rotate_tensor("90")

    def test_rotate_requires_data(self):
        with pytest.raises(ValueError):
            CoreTensor(dim=2).rotate_tensor(0.1)

    def test_rotate_2d_90deg_uniaxial(self):
        # [[1,0],[0,0]] rotated 90deg -> [[0,0],[0,1]]
        data = np.zeros((2, 2, 1))
        data[0, 0, 0] = 1.0
        ct = CoreTensor(data=data, taxis=np.array([0.0]), dim=2)
        ct.rotate_tensor(np.pi / 2)
        npt.assert_allclose(
            ct.data[:, :, 0], np.array([[0.0, 0.0], [0.0, 1.0]]), atol=1e-10
        )

    def test_rotate_3d_around_z_90deg(self):
        # 3D rotation is around Z; uniaxial x stress goes to y.
        data = np.zeros((3, 3, 1))
        data[0, 0, 0] = 1.0
        ct = CoreTensor(data=data, taxis=np.array([0.0]), dim=3)
        ct.rotate_tensor(np.pi / 2)
        expected = np.array(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        )
        npt.assert_allclose(ct.data[:, :, 0], expected, atol=1e-10)

    def test_rotate_unsupported_dimension(self):
        ct = CoreTensor(data=np.zeros((4, 4, 1)), taxis=np.array([0.0]), dim=4)
        with pytest.raises(ValueError):
            ct.rotate_tensor(0.1)

    def test_rotate_diagonal_invariant_under_full_turn(self):
        data = np.zeros((2, 2, 1))
        data[0, 0, 0] = 3.0
        data[1, 1, 0] = -2.0
        ct = CoreTensor(data=data, taxis=np.array([0.0]), dim=2)
        original = ct.data.copy()
        ct.rotate_tensor(2 * np.pi)
        npt.assert_allclose(ct.data, original, atol=1e-10)


# ---------------------------------------------------------------------------
# I/O round-trip
# ---------------------------------------------------------------------------
class TestIO:
    def _make(self, start_time=None, name="io_tensor"):
        data = np.arange(2 * 2 * 3, dtype=float).reshape(2, 2, 3)
        taxis = np.array([0.0, 1.0, 2.0])
        return CoreTensor(
            data=data, taxis=taxis, dim=2, start_time=start_time, name=name
        )

    def test_save_load_roundtrip_with_start_time(self, tmp_path, sample_start_time):
        ct = self._make(start_time=sample_start_time)
        fname = str(tmp_path / "rt.npz")
        ret = ct.savez(fname)
        assert ret is ct
        loaded = CoreTensor().load_npz(fname)
        npt.assert_array_equal(loaded.data, ct.data)
        npt.assert_array_equal(loaded.taxis, ct.taxis)
        assert loaded.dim == 2
        assert loaded.name == "io_tensor"
        assert loaded.start_time == sample_start_time

    def test_save_load_roundtrip_without_start_time(self, tmp_path):
        ct = self._make(start_time=None)
        fname = str(tmp_path / "rt2")  # no extension; auto-appended
        ct.savez(fname)
        loaded = CoreTensor().load_npz(fname)
        assert loaded.start_time is None
        npt.assert_array_equal(loaded.data, ct.data)

    def test_savez_appends_extension(self, tmp_path):
        ct = self._make()
        fname = str(tmp_path / "noext")
        ct.savez(fname)
        assert (tmp_path / "noext.npz").exists()

    def test_savez_requires_essential_attributes(self, tmp_path):
        with pytest.raises(ValueError):
            CoreTensor().savez(str(tmp_path / "empty.npz"))

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            CoreTensor().load_npz("/nonexistent/path/missing.npz")

    def test_load_accepts_path_with_extension(self, tmp_path):
        ct = self._make()
        fname = str(tmp_path / "withext.npz")
        ct.savez(fname)
        loaded = CoreTensor().load_npz(fname)
        assert loaded.dim == 2

    def test_load_missing_required_key_raises_keyerror(self, tmp_path):
        # An .npz that exists but lacks the required 'data'/'taxis'/'dim' keys
        # is re-raised as a KeyError by load_npz.
        fname = str(tmp_path / "bad.npz")
        np.savez(fname, foo=np.array([1, 2, 3]))
        with pytest.raises(KeyError):
            CoreTensor().load_npz(fname)
