# Characterization tests for fiberis.analyzer.Data3D.core3D.Data3D
# and the Data3D_microseismic stub module.
#
# These lock in CURRENT observable behavior to provide a safety net for a
# refactor. Golden values were obtained by running the code, not guessed.
# They must remain GREEN through a pure refactor.
#
# Existing coverage lives in tests/test_core3D.py (initialization, basic
# save/load roundtrip, save validation, info_str). This file deliberately does
# NOT duplicate those and instead characterizes additional behavior.

import os
import sys

import numpy as np
import pytest
from numpy.testing import assert_array_equal

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from fiberis.analyzer.Data3D.core3D import Data3D
from fiberis.analyzer.Data3D import Data3D_microseismic


@pytest.fixture
def small_data3d():
    """A tiny deterministic Data3D: 3 spatial points x 2 time steps."""
    d = Data3D(name="well_A")
    d.data = np.arange(6, dtype=float).reshape(3, 2)
    d.taxis = np.array([0.0, 1.0])
    d.xaxis = np.array([10.0, 20.0, 30.0])
    d.yaxis = np.array([100.0, 200.0, 300.0])
    d.variable_name = "pressure"
    return d


# --------------------------------------------------------------------------- #
# Construction / history                                                       #
# --------------------------------------------------------------------------- #
class TestConstruction:
    def test_default_name_is_none(self):
        d = Data3D()
        assert d.name is None
        assert d.data is None
        assert d.taxis is None
        assert d.xaxis is None
        assert d.yaxis is None
        assert d.variable_name is None

    def test_init_logs_one_history_record(self):
        d = Data3D()
        assert len(d.history.records) == 1
        rec = d.history.records[0]
        assert rec["level"] == "INFO"
        assert "initialized" in rec["description"].lower()

    def test_name_passed_through(self):
        d = Data3D(name="my_sampler")
        assert d.name == "my_sampler"


# --------------------------------------------------------------------------- #
# save/load roundtrips and filename handling                                   #
# --------------------------------------------------------------------------- #
class TestSaveLoad:
    def test_savez_appends_npz_extension(self, small_data3d, tmp_path):
        target = tmp_path / "no_extension"
        small_data3d.savez(str(target))
        # The file written has the .npz suffix appended.
        assert (tmp_path / "no_extension.npz").exists()
        assert not target.exists()

    def test_load_npz_appends_extension(self, small_data3d, tmp_path):
        target = tmp_path / "round"
        small_data3d.savez(str(target))  # writes round.npz

        loaded = Data3D()
        loaded.load_npz(str(target))  # given without extension
        assert_array_equal(loaded.data, small_data3d.data)

    def test_roundtrip_preserves_dtypes_and_values(self, small_data3d, tmp_path):
        fn = tmp_path / "r.npz"
        small_data3d.savez(str(fn))
        loaded = Data3D()
        loaded.load_npz(str(fn))
        assert_array_equal(loaded.data, small_data3d.data)
        assert_array_equal(loaded.taxis, small_data3d.taxis)
        assert_array_equal(loaded.xaxis, small_data3d.xaxis)
        assert_array_equal(loaded.yaxis, small_data3d.yaxis)
        assert loaded.variable_name == "pressure"
        assert loaded.name == "well_A"

    def test_load_npz_stringifies_name_and_variable_name(self, small_data3d, tmp_path):
        # variable_name/name are coerced via str() on load. A genuine None is
        # therefore round-tripped as the literal string 'None'.
        # NOTE: possible bug - None becomes the string 'None' rather than None.
        d = Data3D()  # name is None
        d.data = np.zeros((1, 1))
        d.taxis = np.array([0.0])
        d.xaxis = np.array([0.0])
        d.yaxis = np.array([0.0])
        # variable_name left as None
        fn = tmp_path / "nones.npz"
        d.savez(str(fn))

        loaded = Data3D()
        loaded.load_npz(str(fn))
        assert loaded.name == "None"
        assert isinstance(loaded.name, str)
        assert loaded.variable_name == "None"
        assert isinstance(loaded.variable_name, str)

    def test_load_missing_file_raises_filenotfound(self, tmp_path):
        d = Data3D()
        with pytest.raises(FileNotFoundError):
            d.load_npz(str(tmp_path / "does_not_exist.npz"))
        # An ERROR record is logged before the raise.
        assert any(r["level"] == "ERROR" for r in d.history.records)

    def test_load_npz_missing_key_raises_keyerror(self, tmp_path):
        # An .npz missing a required key (e.g. 'yaxis') raises KeyError and
        # logs an ERROR record.
        fn = tmp_path / "incomplete.npz"
        np.savez(
            str(fn),
            data=np.zeros((1, 1)),
            taxis=np.array([0.0]),
            xaxis=np.array([0.0]),
            variable_name="p",
            name="n",
        )  # no 'yaxis'
        d = Data3D()
        with pytest.raises(KeyError):
            d.load_npz(str(fn))
        assert any(r["level"] == "ERROR" for r in d.history.records)

    def test_savez_records_success_in_history(self, small_data3d, tmp_path):
        before = len(small_data3d.history.records)
        small_data3d.savez(str(tmp_path / "h.npz"))
        assert len(small_data3d.history.records) == before + 1
        assert "saved" in small_data3d.history.records[-1]["description"].lower()


# --------------------------------------------------------------------------- #
# save guard / validation                                                      #
# --------------------------------------------------------------------------- #
class TestSaveValidation:
    @pytest.mark.parametrize("missing", ["data", "taxis", "xaxis", "yaxis"])
    def test_savez_raises_when_any_axis_missing(self, small_data3d, tmp_path, missing):
        setattr(small_data3d, missing, None)
        with pytest.raises(ValueError):
            small_data3d.savez(str(tmp_path / "fail.npz"))

    def test_savez_validation_logs_error(self, tmp_path):
        d = Data3D()
        with pytest.raises(ValueError):
            d.savez(str(tmp_path / "x.npz"))
        assert any(r["level"] == "ERROR" for r in d.history.records)


# --------------------------------------------------------------------------- #
# info / str                                                                   #
# --------------------------------------------------------------------------- #
class TestInfo:
    def test_info_str_unset_object(self):
        s = Data3D().get_info_str()
        assert "Data3D Object Summary: Unnamed" in s
        assert "Name: Not set" in s
        assert "Variable Name: Not set" in s
        assert "Data: Not set" in s
        assert "Time Axis (taxis): Not set" in s
        assert "X Axis (xaxis): Not set" in s
        assert "Y Axis (yaxis): Not set" in s

    def test_info_str_small_axes_show_values_with_ellipsis(self, small_data3d):
        s = small_data3d.get_info_str()
        assert "Data Shape: (3, 2)" in s
        assert "Time Axis (taxis): Length=2" in s
        assert "X Axis (xaxis): Length=3" in s
        assert "Y Axis (yaxis): Length=3" in s
        # Even for short axes, an ellipsis is appended after the values.
        assert "Values (first 10): [10. 20. 30.]..." in s

    def test_info_str_long_axis_truncates_to_first_10(self):
        d = Data3D(name="big")
        d.data = np.zeros((20, 1))
        d.taxis = np.array([0.0])
        d.xaxis = np.arange(20, dtype=float)
        d.yaxis = np.arange(20, dtype=float)
        s = d.get_info_str()
        assert "X Axis (xaxis): Length=20" in s
        # Only the first 10 elements are shown for an axis of size >= 10.
        assert "Values (first 10): [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]..." in s

    def test_str_dunder_matches_get_info_str(self, small_data3d):
        assert str(small_data3d) == small_data3d.get_info_str()

    def test_print_info_runs(self, small_data3d, capsys):
        small_data3d.print_info()
        out = capsys.readouterr().out
        assert "Data3D Object Summary" in out


# --------------------------------------------------------------------------- #
# Data3D_microseismic stub module                                              #
# --------------------------------------------------------------------------- #
class TestMicroseismicStub:
    def test_module_imports_and_is_effectively_empty(self):
        # The microseismic module is currently a placeholder stub with no
        # public classes/functions defined.
        public = [n for n in dir(Data3D_microseismic) if not n.startswith("__")]
        assert public == []
