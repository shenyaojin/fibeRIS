# Characterization tests for fiberis.analyzer.Geometry3D.coreG3D.DataG3D
# and fiberis.analyzer.Geometry3D.DataG3D_md.G3DMeasuredDepth.
#
# These lock in CURRENT observable behavior to provide a safety net for a
# refactor. Golden values were obtained by running the code, not guessed.
#
# Existing coverage lives in tests/test_geometry.py (calculate_md straight/
# diagonal lines, save/load roundtrip, save validation). This file does NOT
# duplicate those and instead characterizes additional behavior: filename
# handling, MD edge cases, info/str, plotting (line/scatter/guards), and the
# measured-depth subclass stubs.

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

from fiberis.analyzer.Geometry3D.coreG3D import DataG3D
from fiberis.analyzer.Geometry3D.DataG3D_md import G3DMeasuredDepth


@pytest.fixture
def straight_well():
    """A straight wellbore along x: x=0..4, y=z=0."""
    t = np.linspace(0, 4, 5)
    g = DataG3D()
    g.xaxis = t.copy()
    g.yaxis = np.zeros_like(t)
    g.zaxis = np.zeros_like(t)
    g.data = np.zeros_like(t)
    g.name = "straight"
    return g


# --------------------------------------------------------------------------- #
# Construction                                                                 #
# --------------------------------------------------------------------------- #
class TestConstruction:
    def test_default_attributes_are_none(self):
        g = DataG3D()
        assert g.data is None
        assert g.xaxis is None
        assert g.yaxis is None
        assert g.zaxis is None
        assert g.name is None

    def test_history_starts_empty(self):
        # Unlike Data3D, DataG3D.__init__ does not log a record.
        g = DataG3D()
        assert g.history.records == []


# --------------------------------------------------------------------------- #
# measured depth                                                               #
# --------------------------------------------------------------------------- #
class TestCalculateMD:
    def test_md_3d_path(self):
        # Path with steps along z too, to exercise the full 3D distance.
        g = DataG3D()
        g.xaxis = np.array([0.0, 3.0, 3.0])
        g.yaxis = np.array([0.0, 0.0, 4.0])
        g.zaxis = np.array([0.0, 0.0, 0.0])
        g.data = np.zeros(3)
        md = g.calculate_md()
        # step1 = 3, step2 = 4 -> cumulative [0, 3, 7]
        assert_allclose(md, [0.0, 3.0, 7.0])

    def test_md_with_z_component(self):
        g = DataG3D()
        g.xaxis = np.array([0.0, 0.0])
        g.yaxis = np.array([0.0, 0.0])
        g.zaxis = np.array([0.0, 5.0])
        g.data = np.zeros(2)
        assert_allclose(g.calculate_md(), [0.0, 5.0])

    def test_md_single_point(self):
        g = DataG3D()
        g.xaxis = np.array([7.0])
        g.yaxis = np.array([2.0])
        g.zaxis = np.array([1.0])
        g.data = np.array([0.0])
        # diff over a single element is empty; result is just [0.0].
        assert_array_equal(g.calculate_md(), np.array([0.0]))

    def test_md_first_value_is_zero(self, straight_well):
        md = straight_well.calculate_md()
        assert md[0] == 0.0
        assert len(md) == len(straight_well.xaxis)

    def test_md_does_not_mutate_axes(self, straight_well):
        x0 = straight_well.xaxis.copy()
        straight_well.calculate_md()
        assert_array_equal(straight_well.xaxis, x0)


# --------------------------------------------------------------------------- #
# save / load                                                                  #
# --------------------------------------------------------------------------- #
class TestSaveLoad:
    def test_savez_appends_extension(self, straight_well, tmp_path):
        target = tmp_path / "noext"
        straight_well.savez(str(target))
        assert (tmp_path / "noext.npz").exists()

    def test_name_after_load_is_basename_without_extension(self, straight_well, tmp_path):
        fn = tmp_path / "well_42.npz"
        straight_well.savez(str(fn))
        loaded = DataG3D()
        loaded.load_npz(str(fn))
        # name is derived from the path: basename minus the 4-char extension.
        assert loaded.name == "well_42"

    def test_load_appends_extension(self, straight_well, tmp_path):
        target = tmp_path / "wname"
        straight_well.savez(str(target))
        loaded = DataG3D()
        loaded.load_npz(str(target))  # no extension supplied
        assert_array_equal(loaded.xaxis, straight_well.xaxis)
        assert loaded.name == "wname"

    def test_roundtrip_preserves_axes(self, straight_well, tmp_path):
        fn = tmp_path / "r.npz"
        straight_well.savez(str(fn))
        loaded = DataG3D()
        loaded.load_npz(str(fn))
        assert_array_equal(loaded.xaxis, straight_well.xaxis)
        assert_array_equal(loaded.yaxis, straight_well.yaxis)
        assert_array_equal(loaded.zaxis, straight_well.zaxis)
        assert_array_equal(loaded.data, straight_well.data)

    @pytest.mark.parametrize("missing", ["data", "xaxis", "yaxis", "zaxis"])
    def test_savez_raises_when_axis_missing(self, straight_well, tmp_path, missing):
        setattr(straight_well, missing, None)
        with pytest.raises(ValueError):
            straight_well.savez(str(tmp_path / "fail.npz"))


# --------------------------------------------------------------------------- #
# info / str                                                                   #
# --------------------------------------------------------------------------- #
class TestInfo:
    def test_info_str_unset(self):
        s = DataG3D().get_info_str()
        assert "DataG3D Object Summary: Unnamed" in s
        assert "Data (MD): Not set" in s
        assert "X Axis (xaxis): Not set" in s
        assert "Y Axis (yaxis): Not set" in s
        assert "Z Axis (zaxis): Not set" in s

    def test_info_str_populated_small_axes(self, straight_well):
        s = straight_well.get_info_str()
        assert "DataG3D Object Summary: straight" in s
        assert "Data (MD): Length=5" in s
        assert "X Axis (xaxis): Length=5" in s
        assert "Z Axis (zaxis): Length=5" in s
        # Short axes still get an ellipsis appended.
        assert "..." in s

    def test_str_dunder_matches_info(self, straight_well):
        assert str(straight_well) == straight_well.get_info_str()

    def test_print_info_runs(self, straight_well, capsys):
        straight_well.print_info()
        assert "DataG3D Object Summary" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# plotting                                                                     #
# --------------------------------------------------------------------------- #
class TestPlot:
    def test_plot_line_returns_list_of_lines(self, straight_well):
        artist = straight_well.plot()
        assert isinstance(artist, list)
        assert isinstance(artist[0], Line2D)
        plt.close("all")

    def test_plot_scatter_on_provided_axes(self, straight_well):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        artist = straight_well.plot(ax=ax, mode="scatter")
        # mpl 3D scatter returns a Path3DCollection.
        assert type(artist).__name__ == "Path3DCollection"
        plt.close("all")

    def test_plot_sets_default_axis_labels(self, straight_well):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        straight_well.plot(ax=ax)
        assert ax.get_xlabel() == "X coordinate"
        assert ax.get_ylabel() == "Y coordinate"
        assert ax.get_zlabel() == "Z coordinate"
        plt.close("all")

    def test_plot_custom_labels_and_title(self, straight_well):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        straight_well.plot(
            ax=ax, xlabel="EW", ylabel="NS", zlabel="TVD", title="My Well"
        )
        assert ax.get_xlabel() == "EW"
        assert ax.get_zlabel() == "TVD"
        assert ax.get_title() == "My Well"
        plt.close("all")

    def test_plot_creates_new_figure_when_ax_none(self, straight_well):
        # With ax=None a new 3D figure is created; default title uses the name.
        n_before = len(plt.get_fignums())
        artist = straight_well.plot()
        assert isinstance(artist[0], Line2D)
        assert len(plt.get_fignums()) == n_before + 1
        plt.close("all")

    def test_plot_invalid_mode_raises(self, straight_well):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        with pytest.raises(ValueError):
            straight_well.plot(ax=ax, mode="bogus")
        plt.close("all")

    def test_plot_guard_when_axes_unset(self):
        g = DataG3D()
        with pytest.raises(ValueError):
            g.plot()
        assert any(r["level"] == "ERROR" for r in g.history.records)

    def test_plot_logs_history_record(self, straight_well):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        before = len(straight_well.history.records)
        straight_well.plot(ax=ax)
        assert len(straight_well.history.records) == before + 1
        plt.close("all")


# --------------------------------------------------------------------------- #
# G3DMeasuredDepth subclass                                                    #
# --------------------------------------------------------------------------- #
class TestG3DMeasuredDepth:
    def test_is_subclass_of_datag3d(self):
        md = G3DMeasuredDepth()
        assert isinstance(md, DataG3D)

    def test_inherits_calculate_md(self):
        md = G3DMeasuredDepth()
        md.xaxis = np.array([0.0, 3.0])
        md.yaxis = np.array([0.0, 4.0])
        md.zaxis = np.array([0.0, 0.0])
        md.data = np.zeros(2)
        assert_allclose(md.calculate_md(), [0.0, 5.0])

    def test_stub_methods_return_none(self):
        # These are currently unimplemented placeholders.
        md = G3DMeasuredDepth()
        assert md.plot_loc() is None
        assert md.get_spatial_coor(123.0) is None
