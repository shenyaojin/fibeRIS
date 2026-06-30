"""Characterization tests for fiberis.analyzer.TensorProcessor.coreT1D.Tensor1D.

History: coreT1D.py previously failed to import because it used ``Tuple[...]``
annotations without importing ``Tuple`` from ``typing``. That defect has been
fixed (see docs/modernization_notes.md, bug #1), so these tests now exercise the
real behavior of the ``Tensor1D`` class instead of pinning the import failure.
"""

import datetime  # noqa: F401  (kept for parity with sibling tensor tests)

import numpy as np
import pytest

from fiberis.analyzer.TensorProcessor.coreT1D import Tensor1D


@pytest.fixture
def sample_tensor():
    """A 4-point field of 2x2 tensors with distinct, easy-to-track components."""
    daxis = np.array([0.0, 1.0, 2.0, 3.0])
    data = np.zeros((4, 2, 2))
    data[:, 0, 0] = [11, 12, 13, 14]  # xx
    data[:, 0, 1] = [21, 22, 23, 24]  # xy
    data[:, 1, 0] = [31, 32, 33, 34]  # yx
    data[:, 1, 1] = [41, 42, 43, 44]  # yy
    t = Tensor1D(name="sample")
    t.set_daxis(daxis).set_data(data).set_dim(2)
    return t


class TestInit:
    def test_default_name_and_history(self):
        t = Tensor1D()
        assert t.name == "Default Tensor1D Data"
        assert t.data is None and t.daxis is None and t.dim is None
        assert len(t.history.records) == 1

    def test_explicit_name(self):
        t = Tensor1D(name="mytensor")
        assert t.name == "mytensor"


class TestSetters:
    def test_set_data_requires_3d(self):
        with pytest.raises(TypeError):
            Tensor1D().set_data(np.zeros((4, 2)))

    def test_set_data_requires_square(self):
        with pytest.raises(ValueError):
            Tensor1D().set_data(np.zeros((4, 2, 3)))

    def test_set_data_infers_dim(self):
        t = Tensor1D().set_data(np.zeros((5, 3, 3)))
        assert t.dim == 3

    def test_set_data_is_defensive_copy(self):
        arr = np.zeros((2, 2, 2))
        t = Tensor1D().set_data(arr)
        arr[0, 0, 0] = 99.0
        assert t.data[0, 0, 0] == 0.0  # copy, not a shared reference

    def test_set_data_daxis_length_mismatch(self):
        t = Tensor1D().set_daxis(np.array([0.0, 1.0]))
        with pytest.raises(ValueError):
            t.set_data(np.zeros((3, 2, 2)))

    def test_set_daxis_requires_1d(self):
        with pytest.raises(TypeError):
            Tensor1D().set_daxis(np.zeros((2, 2)))

    def test_set_dim_rejects_nonpositive(self):
        with pytest.raises(TypeError):
            Tensor1D().set_dim(0)

    def test_set_dim_mismatch_with_data(self):
        t = Tensor1D().set_data(np.zeros((2, 2, 2)))
        with pytest.raises(ValueError):
            t.set_dim(3)

    def test_set_name_type_check(self):
        with pytest.raises(TypeError):
            Tensor1D().set_name(123)

    def test_setters_return_self_for_chaining(self, sample_tensor):
        assert isinstance(sample_tensor, Tensor1D)
        assert sample_tensor.dim == 2


class TestIO:
    def test_savez_requires_complete_state(self, tmp_path):
        with pytest.raises(ValueError):
            Tensor1D().savez(str(tmp_path / "x.npz"))

    def test_savez_load_roundtrip(self, sample_tensor, tmp_path):
        path = str(tmp_path / "tensor")  # extension auto-appended
        sample_tensor.savez(path)
        reloaded = Tensor1D().load_npz(path)
        np.testing.assert_array_equal(reloaded.data, sample_tensor.data)
        np.testing.assert_array_equal(reloaded.daxis, sample_tensor.daxis)
        assert reloaded.dim == 2
        assert reloaded.name == "sample"


class TestGetComponent:
    def test_string_components(self, sample_tensor):
        np.testing.assert_array_equal(sample_tensor.get_component("xx"), [11, 12, 13, 14])
        np.testing.assert_array_equal(sample_tensor.get_component("xy"), [21, 22, 23, 24])
        np.testing.assert_array_equal(sample_tensor.get_component("yx"), [31, 32, 33, 34])
        np.testing.assert_array_equal(sample_tensor.get_component("yy"), [41, 42, 43, 44])

    def test_tuple_component(self, sample_tensor):
        np.testing.assert_array_equal(sample_tensor.get_component((1, 1)), [41, 42, 43, 44])

    def test_invalid_string(self, sample_tensor):
        with pytest.raises(ValueError):
            sample_tensor.get_component("qq")

    def test_out_of_bounds_index(self, sample_tensor):
        with pytest.raises(IndexError):
            sample_tensor.get_component((2, 2))  # dim is 2

    def test_requires_data(self):
        with pytest.raises(ValueError):
            Tensor1D(dim=2).get_component("xx")


class TestPlotAndStr:
    def test_plot_component_default_path(self, sample_tensor):
        # Headless Agg backend is configured in conftest; default call must not raise.
        assert sample_tensor.plot_component("xx") is None

    def test_plot_requires_daxis(self):
        t = Tensor1D()
        t.set_data(np.zeros((2, 2, 2)))
        with pytest.raises(ValueError):
            t.plot_component("xx")

    def test_str_summary(self, sample_tensor):
        s = str(sample_tensor)
        assert "Tensor1D" in s and "(4, 2, 2)" in s and "sample" in s
