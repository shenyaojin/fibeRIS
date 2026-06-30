# Characterization tests for fiberis.analyzer.Data1DG.core1DG.Data1DG.
#
# These lock in CURRENT observable behavior to provide a safety net for a
# refactor. Golden values were obtained by running the code, not guessed, and
# must remain GREEN through a pure refactor.
#
# NOTE: tests/test_geometry.py already exercises basic Data1DG behavior
# (initialization, select_range, shift, get_value_by_location at a node,
# save/load roundtrip). This file does NOT duplicate those; it characterizes
# additional behavior: constructor coercion/validation, interpolation including
# extrapolation clamping, cropping/shift edge cases, copy semantics, info/str,
# plotting, and guard/error paths.

import os
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from fiberis.analyzer.Data1DG.core1DG import Data1DG


@pytest.fixture
def linear_1dg():
    """data == daxis for easy interpolation reasoning: f(x) = x over [0, 10]."""
    daxis = np.linspace(0, 10, 11)
    return Data1DG(data=daxis.copy(), daxis=daxis.copy(),
                   axis_name="Depth (m)", name="lin")


# --------------------------------------------------------------------------- #
# Construction                                                                 #
# --------------------------------------------------------------------------- #
class TestConstruction:
    def test_default_empty_construction(self):
        d = Data1DG()
        assert d.data is None
        assert d.daxis is None
        assert d.axis_name == "Spatial Axis"
        assert d.name is None
        # one record logged for the empty-object path
        assert len(d.history.records) == 1

    def test_inputs_coerced_to_float_arrays(self):
        d = Data1DG(data=[1, 2, 3], daxis=[0, 1, 2])
        assert d.data.dtype == np.dtype(float)
        assert d.daxis.dtype == np.dtype(float)
        assert_array_equal(d.data, [1.0, 2.0, 3.0])

    def test_named_object_logs_named_record(self):
        d = Data1DG(name="abc")
        assert "abc" in d.history.records[0]["description"]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            Data1DG(data=[1.0, 2.0], daxis=[0.0, 1.0, 2.0])

    def test_data_only_no_validation(self):
        # If only one of data/daxis is provided, length check is skipped.
        d = Data1DG(data=[1.0, 2.0, 3.0])
        assert d.daxis is None
        assert_array_equal(d.data, [1.0, 2.0, 3.0])


# --------------------------------------------------------------------------- #
# interpolation                                                                #
# --------------------------------------------------------------------------- #
class TestGetValueByLocation:
    def test_value_at_node(self, linear_1dg):
        assert linear_1dg.get_value_by_location(3.0) == 3.0

    def test_value_interpolated_between_nodes(self, linear_1dg):
        assert_allclose(linear_1dg.get_value_by_location(3.5), 3.5)

    def test_returns_python_float(self, linear_1dg):
        v = linear_1dg.get_value_by_location(2.0)
        assert isinstance(v, float)

    def test_extrapolation_is_clamped_below(self, linear_1dg):
        # np.interp clamps to the first/last value outside the range.
        assert linear_1dg.get_value_by_location(-100.0) == 0.0

    def test_extrapolation_is_clamped_above(self, linear_1dg):
        assert linear_1dg.get_value_by_location(100.0) == 10.0

    def test_raises_when_unset(self):
        with pytest.raises(ValueError):
            Data1DG().get_value_by_location(1.0)

    def test_raises_when_empty(self):
        d = Data1DG(data=np.array([]), daxis=np.array([]))
        with pytest.raises(ValueError):
            d.get_value_by_location(1.0)


# --------------------------------------------------------------------------- #
# select_range / cropping                                                      #
# --------------------------------------------------------------------------- #
class TestSelectRange:
    def test_inclusive_bounds(self, linear_1dg):
        linear_1dg.select_range(2.0, 5.0)
        assert_array_equal(linear_1dg.daxis, [2.0, 3.0, 4.0, 5.0])
        assert_array_equal(linear_1dg.data, [2.0, 3.0, 4.0, 5.0])

    def test_modifies_in_place_returns_none(self, linear_1dg):
        assert linear_1dg.select_range(0.0, 10.0) is None

    def test_empty_selection_yields_empty_arrays(self, linear_1dg):
        linear_1dg.select_range(100.0, 200.0)
        assert linear_1dg.daxis.size == 0
        assert linear_1dg.data.size == 0

    def test_start_greater_than_end_raises(self, linear_1dg):
        with pytest.raises(ValueError):
            linear_1dg.select_range(5.0, 2.0)

    def test_equal_bounds_selects_single_point(self, linear_1dg):
        linear_1dg.select_range(4.0, 4.0)
        assert_array_equal(linear_1dg.daxis, [4.0])

    def test_raises_when_daxis_unset(self):
        with pytest.raises(ValueError):
            Data1DG().select_range(0.0, 1.0)


# --------------------------------------------------------------------------- #
# shift                                                                        #
# --------------------------------------------------------------------------- #
class TestShift:
    def test_positive_shift(self, linear_1dg):
        linear_1dg.shift(5.0)
        assert linear_1dg.daxis[0] == 5.0
        assert linear_1dg.daxis[-1] == 15.0

    def test_negative_shift(self, linear_1dg):
        linear_1dg.shift(-2.0)
        assert linear_1dg.daxis[0] == -2.0

    def test_shift_does_not_touch_data(self, linear_1dg):
        original = linear_1dg.data.copy()
        linear_1dg.shift(3.0)
        assert_array_equal(linear_1dg.data, original)

    def test_shift_guard(self):
        with pytest.raises(ValueError):
            Data1DG().shift(1.0)


# --------------------------------------------------------------------------- #
# copy                                                                         #
# --------------------------------------------------------------------------- #
class TestCopy:
    def test_deepcopy_independence(self, linear_1dg):
        c = linear_1dg.copy()
        c.daxis[0] = 999.0
        assert linear_1dg.daxis[0] == 0.0

    def test_copy_logs_extra_record(self, linear_1dg):
        c = linear_1dg.copy()
        assert len(c.history.records) > len(linear_1dg.history.records)

    def test_copy_preserves_values(self, linear_1dg):
        c = linear_1dg.copy()
        assert_array_equal(c.data, linear_1dg.data)
        assert c.name == linear_1dg.name
        assert c.axis_name == linear_1dg.axis_name


# --------------------------------------------------------------------------- #
# save / load                                                                  #
# --------------------------------------------------------------------------- #
class TestSaveLoad:
    def test_savez_appends_extension(self, linear_1dg, tmp_path):
        target = tmp_path / "noext"
        linear_1dg.savez(str(target))
        assert (tmp_path / "noext.npz").exists()

    def test_roundtrip_preserves_data_and_axis_name(self, linear_1dg, tmp_path):
        fn = tmp_path / "r.npz"
        linear_1dg.savez(str(fn))
        loaded = Data1DG()
        loaded.load_npz(str(fn))
        assert_array_equal(loaded.data, linear_1dg.data)
        assert_array_equal(loaded.daxis, linear_1dg.daxis)
        assert loaded.axis_name == "Depth (m)"

    def test_name_after_load_is_full_basename(self, linear_1dg, tmp_path):
        # Unlike DataG3D, Data1DG keeps the .npz extension in the name.
        fn = tmp_path / "profile.npz"
        linear_1dg.savez(str(fn))
        loaded = Data1DG()
        loaded.load_npz(str(fn))
        assert loaded.name == "profile.npz"

    def test_savez_uses_name_when_no_filename(self, tmp_path):
        # When filename is omitted, self.name is used as the path.
        d = Data1DG(data=[1.0, 2.0], daxis=[0.0, 1.0],
                    name=str(tmp_path / "byname"))
        d.savez()
        assert (tmp_path / "byname.npz").exists()

    def test_savez_no_name_no_filename_raises(self):
        d = Data1DG(data=[1.0], daxis=[0.0])  # name is None
        with pytest.raises(ValueError):
            d.savez()

    def test_savez_raises_when_data_unset(self, tmp_path):
        d = Data1DG()
        with pytest.raises(ValueError):
            d.savez(str(tmp_path / "x.npz"))

    def test_load_missing_file_raises(self, tmp_path):
        d = Data1DG()
        with pytest.raises(FileNotFoundError):
            d.load_npz(str(tmp_path / "nope.npz"))


# --------------------------------------------------------------------------- #
# info / str                                                                   #
# --------------------------------------------------------------------------- #
class TestInfo:
    def test_info_str_unset(self):
        s = Data1DG().get_info_str()
        assert "Data1DG Object Summary: Unnamed" in s
        assert "Spatial Axis (daxis): Not set" in s
        assert "Data: Not set" in s
        assert "Axis Name: Spatial Axis" in s

    def test_info_str_populated_shows_stats(self, linear_1dg):
        s = linear_1dg.get_info_str()
        assert "Data1DG Object Summary: lin" in s
        assert "Axis Name: Depth (m)" in s
        assert "Spatial Axis (daxis): Count=11, Min=0.00, Max=10.00" in s
        assert "Data: Count=11, Min=0.00, Max=10.00" in s

    def test_str_dunder_matches_info(self, linear_1dg):
        assert str(linear_1dg) == linear_1dg.get_info_str()

    def test_print_info_runs(self, linear_1dg, capsys):
        linear_1dg.print_info()
        assert "Data1DG Object Summary" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# plotting                                                                     #
# --------------------------------------------------------------------------- #
class TestPlot:
    def test_plot_returns_list_of_lines(self, linear_1dg):
        fig, ax = plt.subplots()
        lines = linear_1dg.plot(ax=ax)
        assert isinstance(lines, list)
        assert isinstance(lines[0], Line2D)
        plt.close("all")

    def test_plot_sets_axis_labels_from_axis_name(self, linear_1dg):
        fig, ax = plt.subplots()
        linear_1dg.plot(ax=ax)
        assert ax.get_xlabel() == "Depth (m)"
        assert ax.get_ylabel() == "Value"
        plt.close("all")

    def test_plot_default_title_is_name(self, linear_1dg):
        fig, ax = plt.subplots()
        linear_1dg.plot(ax=ax)
        assert ax.get_title() == "lin"
        plt.close("all")

    def test_plot_custom_title(self, linear_1dg):
        fig, ax = plt.subplots()
        linear_1dg.plot(ax=ax, title="Custom")
        assert ax.get_title() == "Custom"
        plt.close("all")

    def test_plot_default_label_uses_name(self, linear_1dg):
        fig, ax = plt.subplots()
        lines = linear_1dg.plot(ax=ax)
        assert lines[0].get_label() == "lin"
        plt.close("all")

    def test_plot_creates_new_figure_when_ax_none(self, linear_1dg):
        # With ax=None a new figure/axes is created and plt.show() is invoked.
        n_before = len(plt.get_fignums())
        lines = linear_1dg.plot()
        assert isinstance(lines[0], Line2D)
        assert len(plt.get_fignums()) == n_before + 1
        plt.close("all")

    def test_plot_guard_when_unset(self):
        with pytest.raises(ValueError):
            Data1DG().plot()

    def test_plot_logs_history_record(self, linear_1dg):
        fig, ax = plt.subplots()
        before = len(linear_1dg.history.records)
        linear_1dg.plot(ax=ax)
        assert len(linear_1dg.history.records) == before + 1
        plt.close("all")
