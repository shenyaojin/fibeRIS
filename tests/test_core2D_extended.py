# Characterization (golden-master) tests for fiberis.analyzer.Data2D.core2D.Data2D
#
# These tests lock in the CURRENT observable behavior of the Data2D base class
# before a refactor. Golden values were obtained by running the code, not by
# guessing. Tests must remain green through a pure refactor.
#
# Existing coverage lives in tests/test_core2D.py (initialization, set_*
# validation, remove_timezone, load/save roundtrip, rename). This file covers
# the remaining surface: cropping, shifting, merging, plotting, value extraction,
# filters, MOOSE export, info/str, copy semantics, and error/guard paths.

import datetime
import os

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
from matplotlib.collections import QuadMesh

from fiberis.analyzer.Data2D.core2D import Data2D


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def synth():
    """Deterministic 4-depth x 5-time Data2D.

    data = 0..19 reshaped (4, 5); taxis = [0,1,2,3,4]; daxis = [10,20,30,40].
    """
    data = np.arange(20, dtype=float).reshape(4, 5)
    taxis = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    daxis = np.array([10.0, 20.0, 30.0, 40.0])
    start_time = datetime.datetime(2023, 1, 1, 12, 0, 0)
    return Data2D(data=data, taxis=taxis, daxis=daxis,
                  start_time=start_time, name="syn")


# --------------------------------------------------------------------------- #
# select_time
# --------------------------------------------------------------------------- #
class TestSelectTime:
    def test_select_time_seconds(self, synth):
        synth.select_time(1.0, 3.0)
        # taxis re-based to start at 0 from the first selected sample.
        assert_array_equal(synth.taxis, np.array([0.0, 1.0, 2.0]))
        assert synth.data.shape == (4, 3)
        # First depth row originally [0,1,2,3,4] -> columns 1..3 -> [1,2,3].
        assert_array_equal(synth.data[0], np.array([1.0, 2.0, 3.0]))
        # start_time advanced by the actual crop offset (1 second).
        assert synth.start_time == datetime.datetime(2023, 1, 1, 12, 0, 1)

    def test_select_time_datetime(self, synth):
        st = synth.start_time
        synth.select_time(st + datetime.timedelta(seconds=1),
                          st + datetime.timedelta(seconds=3))
        assert_array_equal(synth.taxis, np.array([0.0, 1.0, 2.0]))
        assert synth.start_time == datetime.datetime(2023, 1, 1, 12, 0, 1)

    def test_select_time_int_coerced(self, synth):
        synth.select_time(1, 3)
        assert synth.data.shape == (4, 3)

    def test_select_time_empty_range(self, synth):
        # No samples in range: data becomes shape (n_depth, 0), taxis empty,
        # start_time advanced by the *requested* start offset.
        synth.select_time(100, 200)
        assert synth.data.shape == (4, 0)
        assert synth.taxis.size == 0
        assert synth.start_time == datetime.datetime(2023, 1, 1, 12, 1, 40)

    def test_select_time_start_after_end_raises(self, synth):
        with pytest.raises(ValueError):
            synth.select_time(3.0, 1.0)

    def test_select_time_mixed_types_raises(self, synth):
        with pytest.raises(TypeError):
            synth.select_time(1.0, synth.start_time)

    def test_select_time_unset_raises(self):
        with pytest.raises(ValueError):
            Data2D().select_time(0.0, 1.0)


# --------------------------------------------------------------------------- #
# select_depth
# --------------------------------------------------------------------------- #
class TestSelectDepth:
    def test_select_depth_basic(self, synth):
        synth.select_depth(20, 30)
        # daxis is NOT re-based; values preserved.
        assert_array_equal(synth.daxis, np.array([20.0, 30.0]))
        assert synth.data.shape == (2, 5)
        # depth rows 1 and 2 of the original 4x5 matrix.
        assert_array_equal(synth.data[0], np.array([5.0, 6.0, 7.0, 8.0, 9.0]))

    def test_select_depth_empty_range(self, synth):
        synth.select_depth(100, 200)
        assert synth.data.shape == (0, 5)
        assert synth.daxis.size == 0

    def test_select_depth_start_after_end_raises(self, synth):
        with pytest.raises(ValueError):
            synth.select_depth(30, 10)

    def test_select_depth_bad_type_raises(self, synth):
        with pytest.raises(TypeError):
            synth.select_depth("a", "b")

    def test_select_depth_unset_raises(self):
        with pytest.raises(ValueError):
            Data2D().select_depth(0, 1)


# --------------------------------------------------------------------------- #
# shift
# --------------------------------------------------------------------------- #
class TestShift:
    def test_shift_seconds(self, synth):
        synth.shift(5)
        assert synth.start_time == datetime.datetime(2023, 1, 1, 12, 0, 5)

    def test_shift_timedelta(self, synth):
        synth.shift(datetime.timedelta(seconds=2))
        assert synth.start_time == datetime.datetime(2023, 1, 1, 12, 0, 2)

    def test_shift_negative(self, synth):
        synth.shift(-1.5)
        assert synth.start_time == datetime.datetime(2023, 1, 1, 11, 59, 58, 500000)

    def test_shift_no_start_time_raises(self):
        with pytest.raises(ValueError):
            Data2D(data=np.ones((2, 2)), taxis=np.arange(2.0),
                   daxis=np.arange(2.0)).shift(1.0)

    def test_shift_bad_type_raises(self, synth):
        with pytest.raises(TypeError):
            synth.shift("1s")


# --------------------------------------------------------------------------- #
# right_merge
# --------------------------------------------------------------------------- #
class TestRightMerge:
    def test_merge_basic(self, synth):
        other = synth.copy()
        other.start_time = synth.start_time + datetime.timedelta(seconds=10)
        synth.right_merge(other)
        assert_array_equal(
            synth.taxis,
            np.array([0.0, 1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0, 14.0]),
        )
        assert synth.data.shape == (4, 10)

    def test_merge_overlap_raises(self, synth):
        other = synth.copy()
        other.start_time = synth.start_time + datetime.timedelta(seconds=1)
        with pytest.raises(ValueError):
            synth.right_merge(other)

    def test_merge_daxis_mismatch_raises(self, synth):
        other = synth.copy()
        other.daxis = np.array([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError):
            synth.right_merge(other)

    def test_merge_wrong_type_raises(self, synth):
        with pytest.raises(TypeError):
            synth.right_merge("not a Data2D")

    def test_merge_into_empty_copies_other(self, synth):
        empty = Data2D(data=np.empty((4, 0)), taxis=np.array([]),
                       daxis=synth.daxis.copy(),
                       start_time=synth.start_time, name="empty")
        empty.right_merge(synth)
        assert_array_equal(empty.taxis, synth.taxis)
        assert empty.data.shape == (4, 5)

    def test_merge_other_empty_noop(self, synth):
        other = Data2D(data=np.empty((4, 0)), taxis=np.array([]),
                       daxis=synth.daxis.copy(),
                       start_time=synth.start_time + datetime.timedelta(seconds=10),
                       name="empty")
        synth.right_merge(other)
        assert synth.data.shape == (4, 5)

    def test_merge_missing_attrs_raises(self, synth):
        with pytest.raises(ValueError):
            synth.right_merge(Data2D())


# --------------------------------------------------------------------------- #
# value extraction
# --------------------------------------------------------------------------- #
class TestValueExtraction:
    def test_get_value_by_time_seconds(self, synth):
        vals, t = synth.get_value_by_time(2.0)
        assert t == 2.0
        # Column 2 of the matrix: [2, 7, 12, 17].
        assert_array_equal(vals, np.array([2.0, 7.0, 12.0, 17.0]))

    def test_get_value_by_time_datetime(self, synth):
        vals, t = synth.get_value_by_time(
            synth.start_time + datetime.timedelta(seconds=3))
        assert t == 3.0
        assert_array_equal(vals, np.array([3.0, 8.0, 13.0, 18.0]))

    def test_get_value_by_time_empty_taxis_raises(self, synth):
        empty = Data2D(data=np.empty((4, 0)), taxis=np.array([]),
                       daxis=synth.daxis.copy(), start_time=synth.start_time)
        with pytest.raises(ValueError):
            empty.get_value_by_time(1.0)

    def test_get_value_by_time_bad_type_raises(self, synth):
        with pytest.raises(TypeError):
            synth.get_value_by_time("now")

    def test_get_value_by_depth_nearest(self, synth):
        # depth 25 -> nearest channel is daxis index 1 (value 20).
        out = synth.get_value_by_depth(25.0)
        assert_array_equal(out, np.array([5.0, 6.0, 7.0, 8.0, 9.0]))

    def test_get_value_by_depth_out_of_range_returns_none(self, synth):
        assert synth.get_value_by_depth(999.0) is None

    def test_get_value_by_depth_empty_daxis_returns_none(self, synth):
        empty = Data2D(data=np.empty((0, 5)), taxis=synth.taxis.copy(),
                       daxis=np.array([]), start_time=synth.start_time)
        assert empty.get_value_by_depth(1.0) is None

    def test_get_max_axes(self, synth):
        assert synth.get_max_daxis() == 40.0
        assert synth.get_max_taxis() == 4.0

    def test_get_max_axes_none_when_unset(self):
        assert Data2D().get_max_daxis() is None
        assert Data2D().get_max_taxis() is None


# --------------------------------------------------------------------------- #
# time helpers
# --------------------------------------------------------------------------- #
class TestTimeHelpers:
    def test_get_start_time(self, synth):
        assert synth.get_start_time() == datetime.datetime(2023, 1, 1, 12, 0, 0)

    def test_get_end_time_datetime(self, synth):
        assert synth.get_end_time() == datetime.datetime(2023, 1, 1, 12, 0, 4)

    def test_get_end_time_seconds(self, synth):
        assert synth.get_end_time("seconds") == 4.0

    def test_get_end_time_bad_format_raises(self, synth):
        with pytest.raises(ValueError):
            synth.get_end_time("bogus")

    def test_get_end_time_none_when_unset(self):
        assert Data2D().get_end_time() is None

    def test_calculate_time(self, synth):
        out = synth.calculate_time()
        expected = np.array(
            ["2023-01-01T12:00:00", "2023-01-01T12:00:01", "2023-01-01T12:00:02",
             "2023-01-01T12:00:03", "2023-01-01T12:00:04"],
            dtype="datetime64[s]",
        )
        assert_array_equal(out.astype("datetime64[s]"), expected)

    def test_calculate_time_no_start_raises(self):
        d = Data2D(taxis=np.arange(3.0))
        with pytest.raises(ValueError):
            d.calculate_time()

    def test_calculate_time_seconds_returns_taxis(self, synth):
        assert synth.calculate_time_seconds() is synth.taxis


# --------------------------------------------------------------------------- #
# filters
# --------------------------------------------------------------------------- #
class TestFilters:
    @pytest.fixture
    def noisy(self):
        rng = np.random.default_rng(0)
        data = rng.random((3, 100))
        taxis = np.linspace(0, 1, 100)
        daxis = np.arange(3.0)
        return Data2D(data=data, taxis=taxis, daxis=daxis,
                      start_time=datetime.datetime(2023, 1, 1))

    def test_lowpass_preserves_shape(self, noisy):
        noisy.apply_lowpass_filter(cutoff_freq=10.0)
        assert noisy.data.shape == (3, 100)

    def test_lowpass_explicit_sample_rate(self, noisy):
        noisy.apply_lowpass_filter(cutoff_freq=10.0, sample_rate=100.0, order=3)
        assert noisy.data.shape == (3, 100)

    def test_bandpass_preserves_shape(self, noisy):
        noisy.apply_bandpass_filter(lowcut_freq=5.0, highcut_freq=20.0)
        assert noisy.data.shape == (3, 100)

    def test_lowpass_short_taxis_skips(self):
        d = Data2D(data=np.zeros((2, 1)), taxis=np.array([0.0]),
                   daxis=np.arange(2.0), start_time=datetime.datetime(2023, 1, 1))
        d.apply_lowpass_filter(5.0)  # no raise, no-op
        assert d.data.shape == (2, 1)

    def test_lowpass_unset_raises(self):
        with pytest.raises(ValueError):
            Data2D().apply_lowpass_filter(5.0)

    def test_bandpass_unset_raises(self):
        with pytest.raises(ValueError):
            Data2D().apply_bandpass_filter(1.0, 5.0)


# --------------------------------------------------------------------------- #
# plotting (Agg backend; assert no-raise / return types)
# --------------------------------------------------------------------------- #
class TestPlotting:
    def teardown_method(self):
        plt.close("all")

    def test_plot_pcolormesh(self, synth):
        fig, ax = plt.subplots()
        art = synth.plot(ax=ax, method="pcolormesh")
        assert isinstance(art, QuadMesh)

    def test_plot_imshow(self, synth):
        fig, ax = plt.subplots()
        art = synth.plot(ax=ax, method="imshow")
        assert isinstance(art, AxesImage)

    def test_plot_imshow_timestamp(self, synth):
        fig, ax = plt.subplots()
        art = synth.plot(ax=ax, method="imshow", use_timestamp=True)
        assert isinstance(art, AxesImage)

    def test_plot_colorbar_and_clim(self, synth):
        fig, ax = plt.subplots()
        art = synth.plot(ax=ax, method="pcolormesh", colorbar=True,
                         clim=(0.0, 10.0), clabel="amp")
        assert isinstance(art, QuadMesh)

    def test_plot_invalid_method_raises(self, synth):
        fig, ax = plt.subplots()
        with pytest.raises(ValueError):
            synth.plot(ax=ax, method="bogus")

    def test_plot_unset_data_raises(self):
        with pytest.raises(ValueError):
            Data2D().plot()

    def test_plot_empty_returns_none(self, synth):
        synth.select_time(100, 200)  # makes data empty along time axis
        fig, ax = plt.subplots()
        assert synth.plot(ax=ax) is None

    def test_plot_legacy_usetimestamp_kwarg(self, synth):
        fig, ax = plt.subplots()
        art = synth.plot(ax=ax, method="imshow", useTimeStamp=True)
        assert isinstance(art, AxesImage)


# --------------------------------------------------------------------------- #
# info / str
# --------------------------------------------------------------------------- #
class TestInfoStr:
    def test_get_info_str_populated(self, synth):
        s = synth.get_info_str()
        assert "Data2D Object Summary: syn" in s
        assert "Data Shape: (4, 5)" in s
        assert "Time Axis (taxis): Length=5" in s
        assert "Depth Axis (daxis): Length=4" in s

    def test_str_equals_info_str(self, synth):
        assert str(synth) == synth.get_info_str()

    def test_info_str_empty(self):
        s = Data2D().get_info_str()
        assert "Unnamed" in s
        assert "Data: Not set" in s
        assert "Time Axis (taxis): Not set" in s

    def test_info_str_long_axis_truncates(self):
        d = Data2D(data=np.zeros((20, 20)), taxis=np.arange(20.0),
                   daxis=np.arange(20.0))
        s = d.get_info_str()
        assert "first 10" in s

    def test_print_info_runs(self, synth, capsys):
        synth.print_info()
        out = capsys.readouterr().out
        assert "Data2D Object Summary" in out


# --------------------------------------------------------------------------- #
# MOOSE reporter export
# --------------------------------------------------------------------------- #
class TestMooseReporter:
    def test_basic_structure(self, synth):
        out = synth.to_moose_reporter_str(coord_x=1.0, coord_y=2.0)
        assert "measurement_points" in out
        assert "measurement_time" in out
        assert "measurement_values" in out
        # 4 depths x 5 times = 20 measurements.
        assert "Total measurements: 20" in out

    def test_values_are_fortran_flatten(self, synth):
        out = synth.to_moose_reporter_str(coord_x=0.0, coord_y=0.0, precision=1)
        values_line = [ln for ln in out.splitlines()
                       if "measurement_values" in ln][0]
        # Fortran-order flatten of arange(20).reshape(4,5): first column 0,5,10,15.
        assert "0.0   5.0   10.0   15.0" in values_line

    def test_requires_exactly_one_none_coord(self, synth):
        with pytest.raises(ValueError):
            synth.to_moose_reporter_str(coord_x=1, coord_y=2, coord_z=3)
        with pytest.raises(ValueError):
            synth.to_moose_reporter_str(coord_x=1)  # two None

    def test_unset_data_raises(self):
        with pytest.raises(ValueError):
            Data2D().to_moose_reporter_str(coord_x=1.0, coord_y=2.0)


# --------------------------------------------------------------------------- #
# copy semantics
# --------------------------------------------------------------------------- #
class TestCopy:
    def test_deepcopy_independent(self):
        a = Data2D(data=np.ones((2, 2)), taxis=np.arange(2.0),
                   daxis=np.arange(2.0), name="c")
        b = a.copy()
        b.data[0, 0] = 99.0
        assert a.data[0, 0] == 1.0  # deep copy is independent

    def test_shallow_copy_shares_data(self):
        import copy as _copy
        a = Data2D(data=np.ones((2, 2)), taxis=np.arange(2.0),
                   daxis=np.arange(2.0), name="c")
        c = _copy.copy(a)
        c.data[0, 0] = 77.0
        # NOTE: __copy__ shares the underlying ndarray (shallow copy semantics).
        assert a.data[0, 0] == 77.0

    def test_copy_has_independent_history(self, synth):
        b = synth.copy()
        n_before = len(synth.history.records)
        b.history.add_record("extra")
        assert len(synth.history.records) == n_before


# --------------------------------------------------------------------------- #
# I/O: savez roundtrip + load error paths
# --------------------------------------------------------------------------- #
class TestIO:
    def test_savez_load_roundtrip(self, synth, tmp_path):
        fn = str(tmp_path / "out")  # no extension; savez appends .npz
        synth.savez(fn)
        assert os.path.exists(fn + ".npz")
        loaded = Data2D()
        loaded.load_npz(fn)
        assert_allclose(loaded.data, synth.data)
        assert_allclose(loaded.taxis, synth.taxis)
        assert_allclose(loaded.daxis, synth.daxis)
        assert loaded.start_time == synth.start_time
        assert loaded.name == "out.npz"

    def test_savez_without_data_raises(self, tmp_path):
        with pytest.raises(ValueError):
            Data2D().savez(str(tmp_path / "x"))

    def test_load_npz_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            Data2D().load_npz("/no/such/file.npz")

    def test_load_real_example(self, examples_data_dir):
        path = os.path.join(examples_data_dir, "2d", "fiberis_format",
                            "DASdata_example.npz")
        d = Data2D()
        d.load_npz(path)
        # Real DAS dataset: data converted to float, axes wired up.
        assert d.data.shape == (6100, 258)
        assert d.data.dtype == np.float64
        assert d.taxis.shape == (258,)
        assert d.daxis.shape == (6100,)
        assert d.name == "DASdata_example.npz"
        # start_time parsed from tz-aware datetime64/object in the file.
        assert d.start_time.year == 2019
        assert d.start_time.month == 3
        assert d.start_time.day == 28


# --------------------------------------------------------------------------- #
# misc setters / logging compatibility
# --------------------------------------------------------------------------- #
class TestMisc:
    def test_record_log_and_history(self, synth):
        n_before = len(synth.history.records)
        synth.record_log("hello", "world", level="WARNING")
        assert len(synth.history.records) == n_before + 1

    def test_print_log_runs(self, synth):
        synth.print_log()  # no raise

    def test_set_filename_joins(self, synth):
        synth.set_filename("a", "b", "c")
        assert synth.name == "a_b_c"

    def test_set_name_bad_type_raises(self, synth):
        with pytest.raises(TypeError):
            synth.set_name(123)
