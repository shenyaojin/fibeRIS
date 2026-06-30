# Characterization (golden-master) tests for the UNTESTED methods of
# fiberis.analyzer.Data1D.core1D.Data1D.
#
# These tests pin the CURRENT observable behavior of the code so that a pure
# refactor can be validated against them. Golden values were obtained by
# actually running the code, not by reasoning about what is "correct".
# Where behavior looks like a possible bug, the test still pins the current
# behavior and a NOTE comment flags it.

import datetime
import os

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from matplotlib.lines import Line2D

from fiberis.analyzer.Data1D.core1D import Data1D


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def linear_data(sample_start_time):
    """A small, monotonic, easy-to-reason-about Data1D instance.

    taxis = [0, 1, 2, 3, 4]; data = [0, 10, 20, 30, 40].
    """
    taxis = np.arange(5, dtype=float)
    data = taxis * 10.0
    return Data1D(data=data.copy(), taxis=taxis.copy(),
                  start_time=sample_start_time, name="linear")


@pytest.fixture
def sine_obj(sine_taxis_data, sample_start_time):
    taxis, data = sine_taxis_data
    return Data1D(data=data.copy(), taxis=taxis.copy(),
                  start_time=sample_start_time, name="sine")


# ---------------------------------------------------------------------------
# load_npz
# ---------------------------------------------------------------------------
class TestLoadNpz:
    def test_load_npz_with_datetime_start_time(self, tmp_path):
        fn = tmp_path / "dt.npz"
        data = np.array([1, 2, 3])  # ints on purpose -> should become float
        taxis = np.array([0, 1, 2])
        st = datetime.datetime(2023, 5, 6, 7, 8, 9)
        np.savez(fn, data=data, taxis=taxis, start_time=st)

        d = Data1D()
        d.load_npz(str(fn))

        assert d.data.dtype == float
        assert d.taxis.dtype == float
        assert_array_equal(d.data, np.array([1.0, 2.0, 3.0]))
        assert d.start_time == st
        assert d.name == "dt.npz"

    def test_load_npz_with_iso_string_start_time(self, tmp_path):
        fn = tmp_path / "iso.npz"
        np.savez(fn, data=np.array([1.0]), taxis=np.array([0.0]),
                 start_time="2025-01-01T04:51:04")

        d = Data1D()
        d.load_npz(str(fn))
        assert d.start_time == datetime.datetime(2025, 1, 1, 4, 51, 4)

    def test_load_npz_real_example_file(self, examples_data_dir):
        fn = os.path.join(examples_data_dir, "1d", "001_pressure_data_sample.npz")
        d = Data1D()
        d.load_npz(fn)
        assert d.data.size == 96
        assert d.taxis.size == 96
        assert d.name == "001_pressure_data_sample.npz"
        assert isinstance(d.start_time, datetime.datetime)

    def test_load_npz_file_not_found(self):
        d = Data1D()
        with pytest.raises(FileNotFoundError):
            d.load_npz("/nonexistent/path/to/file.npz")

    def test_load_npz_missing_keys(self, tmp_path):
        fn = tmp_path / "missing.npz"
        np.savez(fn, data=np.array([1.0]))  # no taxis / start_time
        d = Data1D()
        with pytest.raises(KeyError):
            d.load_npz(str(fn))

    def test_load_npz_unsupported_start_time_type(self, tmp_path):
        fn = tmp_path / "badtype.npz"
        # start_time stored as an int -> not str and not datetime
        np.savez(fn, data=np.array([1.0]), taxis=np.array([0.0]),
                 start_time=np.array(12345))
        d = Data1D()
        with pytest.raises(ValueError):
            d.load_npz(str(fn))


# ---------------------------------------------------------------------------
# crop / select_time
# ---------------------------------------------------------------------------
class TestCrop:
    def test_crop_requires_start_time(self):
        d = Data1D(data=np.arange(3.0), taxis=np.arange(3.0))
        with pytest.raises(ValueError):
            d.crop(0.0, 1.0)

    def test_crop_requires_data(self, sample_start_time):
        d = Data1D(start_time=sample_start_time)
        with pytest.raises(ValueError):
            d.crop(0.0, 1.0)

    def test_crop_invalid_type_raises_typeerror(self, linear_data):
        with pytest.raises(TypeError):
            linear_data.crop("a", "b")

    def test_crop_int_inputs_normalized(self, linear_data):
        # ints are accepted and treated as seconds
        linear_data.crop(1, 3)
        assert_array_equal(linear_data.taxis, np.array([0.0, 1.0, 2.0]))
        assert_array_equal(linear_data.data, np.array([10.0, 20.0, 30.0]))

    def test_crop_on_empty_taxis_shifts_start_time(self, sample_start_time):
        d = Data1D(data=np.array([]), taxis=np.array([]),
                   start_time=sample_start_time)
        d.crop(2.0, 5.0)
        # start_time advanced by start_seconds (2.0), data stays empty
        assert d.start_time == sample_start_time + datetime.timedelta(seconds=2.0)
        assert d.taxis.size == 0

    def test_crop_empty_result_shifts_start_time(self, linear_data, sample_start_time):
        linear_data.crop(100.0, 200.0)
        assert linear_data.data.size == 0
        assert linear_data.taxis.size == 0
        # start_time advanced by requested start_seconds (100)
        assert linear_data.start_time == sample_start_time + datetime.timedelta(seconds=100.0)

    def test_select_time_is_alias_for_crop(self, linear_data):
        assert Data1D.select_time is Data1D.crop
        linear_data.select_time(1.0, 3.0)
        assert_array_equal(linear_data.taxis, np.array([0.0, 1.0, 2.0]))

    def test_crop_datetime_clamps_negative_start_to_zero(self, linear_data, sample_start_time):
        # start before start_time -> start_seconds clamped to 0
        start = sample_start_time - datetime.timedelta(seconds=10)
        end = sample_start_time + datetime.timedelta(seconds=2)
        linear_data.crop(start, end)
        assert_array_equal(linear_data.taxis, np.array([0.0, 1.0, 2.0]))
        assert linear_data.start_time == sample_start_time


# ---------------------------------------------------------------------------
# shift
# ---------------------------------------------------------------------------
class TestShift:
    def test_shift_seconds(self, linear_data, sample_start_time):
        linear_data.shift(5.0)
        assert linear_data.start_time == sample_start_time + datetime.timedelta(seconds=5.0)

    def test_shift_timedelta(self, linear_data, sample_start_time):
        linear_data.shift(datetime.timedelta(minutes=1))
        assert linear_data.start_time == sample_start_time + datetime.timedelta(minutes=1)

    def test_shift_negative_int(self, linear_data, sample_start_time):
        linear_data.shift(-3)
        assert linear_data.start_time == sample_start_time - datetime.timedelta(seconds=3)

    def test_shift_requires_start_time(self):
        d = Data1D(data=np.arange(3.0), taxis=np.arange(3.0))
        with pytest.raises(ValueError):
            d.shift(1.0)

    def test_shift_invalid_type(self, linear_data):
        with pytest.raises(TypeError):
            linear_data.shift("nope")


# ---------------------------------------------------------------------------
# get_value_by_time
# ---------------------------------------------------------------------------
class TestGetValueByTime:
    def test_float_interpolation(self, linear_data):
        assert linear_data.get_value_by_time(1.5) == 15.0

    def test_int_input(self, linear_data):
        assert linear_data.get_value_by_time(2) == 20.0

    def test_datetime_input(self, linear_data, sample_start_time):
        tp = sample_start_time + datetime.timedelta(seconds=1)
        assert linear_data.get_value_by_time(tp) == 10.0

    def test_extrapolation_clamps_to_endpoints(self, linear_data):
        # np.interp clamps outside the range rather than extrapolating
        assert linear_data.get_value_by_time(-100.0) == 0.0
        assert linear_data.get_value_by_time(100.0) == 40.0

    def test_single_point_exact(self, sample_start_time):
        d = Data1D(data=np.array([5.0]), taxis=np.array([2.0]),
                   start_time=sample_start_time)
        assert d.get_value_by_time(2.0) == 5.0

    def test_single_point_off_returns_the_point(self, sample_start_time):
        # NOTE: documents current behavior; np.interp on a single point returns
        # that point's value for any query time (no true extrapolation).
        d = Data1D(data=np.array([5.0]), taxis=np.array([2.0]),
                   start_time=sample_start_time)
        assert d.get_value_by_time(100.0) == 5.0

    def test_empty_data_raises(self, sample_start_time):
        d = Data1D(data=np.array([]), taxis=np.array([]),
                   start_time=sample_start_time)
        with pytest.raises(ValueError):
            d.get_value_by_time(1.0)

    def test_datetime_without_start_time_raises(self):
        d = Data1D(data=np.array([1.0, 2.0]), taxis=np.array([0.0, 1.0]))
        with pytest.raises(ValueError):
            d.get_value_by_time(datetime.datetime(2023, 1, 1))

    def test_invalid_type_raises(self, linear_data):
        with pytest.raises(TypeError):
            linear_data.get_value_by_time("nope")


# ---------------------------------------------------------------------------
# calculate_time
# ---------------------------------------------------------------------------
class TestCalculateTime:
    def test_returns_datetime64_array(self, linear_data, sample_start_time):
        out = linear_data.calculate_time()
        assert out.size == 5
        assert np.issubdtype(out.dtype, np.datetime64)
        assert out[0] == np.datetime64(sample_start_time)
        assert out[-1] == np.datetime64(sample_start_time) + np.timedelta64(4, "s")

    def test_requires_start_time(self):
        d = Data1D(taxis=np.arange(3.0))
        with pytest.raises(ValueError):
            d.calculate_time()

    def test_requires_taxis(self, sample_start_time):
        d = Data1D(start_time=sample_start_time)
        with pytest.raises(ValueError):
            d.calculate_time()


# ---------------------------------------------------------------------------
# copy
# ---------------------------------------------------------------------------
class TestCopy:
    def test_copy_is_deep(self, linear_data):
        c = linear_data.copy()
        assert c is not linear_data
        assert c.data is not linear_data.data
        c.data[0] = 999.0
        assert linear_data.data[0] == 0.0

    def test_copy_preserves_attrs(self, linear_data):
        c = linear_data.copy()
        assert c.name == linear_data.name
        assert c.start_time == linear_data.start_time
        assert_array_equal(c.taxis, linear_data.taxis)

    def test_copy_adds_history_record(self, linear_data):
        c = linear_data.copy()
        assert len(c.history.records) > len(linear_data.history.records)


# ---------------------------------------------------------------------------
# rename
# ---------------------------------------------------------------------------
class TestRename:
    def test_rename_ok(self, linear_data):
        linear_data.rename("renamed")
        assert linear_data.name == "renamed"

    def test_rename_strips_whitespace(self, linear_data):
        linear_data.rename("  padded  ")
        assert linear_data.name == "padded"

    def test_rename_non_string_raises_typeerror(self, linear_data):
        with pytest.raises(TypeError):
            linear_data.rename(123)

    def test_rename_empty_raises_valueerror(self, linear_data):
        with pytest.raises(ValueError):
            linear_data.rename("   ")


# ---------------------------------------------------------------------------
# down_sample
# ---------------------------------------------------------------------------
class TestDownSample:
    def test_down_sample_factor_2(self, sample_start_time):
        d = Data1D(data=np.arange(10.0), taxis=np.arange(10.0),
                   start_time=sample_start_time)
        d.down_sample(2)
        assert_array_equal(d.data, np.array([0.0, 2.0, 4.0, 6.0, 8.0]))
        assert_array_equal(d.taxis, np.array([0.0, 2.0, 4.0, 6.0, 8.0]))

    def test_down_sample_factor_1_noop(self, sample_start_time):
        d = Data1D(data=np.arange(5.0), taxis=np.arange(5.0),
                   start_time=sample_start_time)
        d.down_sample(1)
        assert d.data.size == 5

    def test_down_sample_zero_raises(self, sample_start_time):
        d = Data1D(data=np.arange(5.0), taxis=np.arange(5.0),
                   start_time=sample_start_time)
        with pytest.raises(ValueError):
            d.down_sample(0)

    def test_down_sample_negative_raises(self, sample_start_time):
        d = Data1D(data=np.arange(5.0), taxis=np.arange(5.0),
                   start_time=sample_start_time)
        with pytest.raises(ValueError):
            d.down_sample(-2)

    def test_down_sample_float_raises(self, sample_start_time):
        d = Data1D(data=np.arange(5.0), taxis=np.arange(5.0),
                   start_time=sample_start_time)
        with pytest.raises(ValueError):
            d.down_sample(2.5)

    def test_down_sample_empty_data_noop(self, sample_start_time):
        d = Data1D(data=np.array([]), taxis=np.array([]),
                   start_time=sample_start_time)
        d.down_sample(2)  # warning + return, no raise
        assert d.data.size == 0

    def test_down_sample_no_data_raises(self):
        d = Data1D()
        with pytest.raises(ValueError):
            d.down_sample(2)


# ---------------------------------------------------------------------------
# get_end_time
# ---------------------------------------------------------------------------
class TestGetEndTime:
    def test_timestamp(self, linear_data, sample_start_time):
        end = linear_data.get_end_time(use_timestamp=True)
        assert end == sample_start_time + datetime.timedelta(seconds=4.0)

    def test_seconds(self, linear_data):
        end = linear_data.get_end_time(use_timestamp=False)
        assert end == 4.0
        assert isinstance(end, np.float64)

    def test_requires_start_time(self):
        d = Data1D(taxis=np.arange(3.0))
        with pytest.raises(ValueError):
            d.get_end_time()

    def test_empty_taxis_raises(self, sample_start_time):
        d = Data1D(taxis=np.array([]), start_time=sample_start_time)
        with pytest.raises(ValueError):
            d.get_end_time()


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------
class TestPlot:
    def test_plot_returns_line2d_list(self, sine_obj):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        lines = sine_obj.plot(ax=ax)
        assert isinstance(lines, list)
        assert len(lines) == 1
        assert isinstance(lines[0], Line2D)
        plt.close(fig)

    def test_plot_with_timestamp(self, sine_obj):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        lines = sine_obj.plot(ax=ax, use_timestamp=True, title="t")
        assert len(lines) == 1
        plt.close(fig)

    def test_plot_no_axes_creates_figure(self, sine_obj):
        import matplotlib.pyplot as plt
        lines = sine_obj.plot()  # ax=None -> new fig, plt.show() (Agg no-op)
        assert isinstance(lines, list) and len(lines) == 1
        plt.close("all")

    def test_plot_empty_with_axes_returns_empty_list(self, sample_start_time):
        import matplotlib.pyplot as plt
        d = Data1D(data=np.array([]), taxis=np.array([]),
                   start_time=sample_start_time)
        fig, ax = plt.subplots()
        lines = d.plot(ax=ax)
        assert lines == []
        plt.close(fig)

    def test_plot_no_data_raises(self):
        d = Data1D()
        with pytest.raises(ValueError):
            d.plot()


# ---------------------------------------------------------------------------
# get_info_str / print_info / print_str / __str__
# ---------------------------------------------------------------------------
class TestInfoStrings:
    def test_get_info_str_contains_name_and_lengths(self, linear_data):
        s = linear_data.get_info_str()
        assert "linear" in s
        assert "Length=5" in s
        assert "Start Time:" in s

    def test_get_info_str_long_arrays_truncated(self, sample_start_time):
        d = Data1D(data=np.arange(20.0), taxis=np.arange(20.0),
                   start_time=sample_start_time, name="long")
        s = d.get_info_str()
        assert "first 10" in s

    def test_get_info_str_unset_fields(self):
        d = Data1D()
        s = d.get_info_str()
        assert "Unnamed" in s
        assert "Not set" in s

    def test_str_equals_get_info_str(self, linear_data):
        assert str(linear_data) == linear_data.get_info_str()

    def test_print_info(self, linear_data, capsys):
        linear_data.print_info()
        out = capsys.readouterr().out
        assert "linear" in out

    def test_print_str(self, linear_data, capsys):
        linear_data.print_str()
        out = capsys.readouterr().out
        # both taxis line and data line printed
        assert "0.0" in out and "40.0" in out

    def test_print_str_unset(self, capsys):
        d = Data1D()
        d.print_str()
        out = capsys.readouterr().out
        assert "not set" in out.lower()


# ---------------------------------------------------------------------------
# right_merge
# ---------------------------------------------------------------------------
class TestRightMerge:
    def test_normal_merge(self, sample_start_time):
        a = Data1D(data=np.arange(5.0), taxis=np.arange(5.0),
                   start_time=sample_start_time, name="a")
        b = Data1D(data=np.arange(3.0), taxis=np.arange(3.0),
                   start_time=sample_start_time + datetime.timedelta(seconds=15),
                   name="b")
        a.right_merge(b)
        assert a.data.size == 8
        # other taxis offset by 15s relative to a's start
        assert a.taxis[5] == 15.0
        assert a.taxis[-1] == 17.0

    def test_empty_self_copies_other(self, sample_start_time):
        a = Data1D(data=np.array([]), taxis=np.array([]),
                   start_time=sample_start_time, name="a")
        b = Data1D(data=np.arange(3.0), taxis=np.arange(3.0),
                   start_time=sample_start_time + datetime.timedelta(seconds=2),
                   name="b")
        a.right_merge(b)
        assert a.data.size == 3
        # self adopts other's start_time
        assert a.start_time == sample_start_time + datetime.timedelta(seconds=2)

    def test_empty_other_is_noop(self, sample_start_time):
        a = Data1D(data=np.arange(5.0), taxis=np.arange(5.0),
                   start_time=sample_start_time, name="a")
        empty = Data1D(data=np.array([]), taxis=np.array([]),
                       start_time=sample_start_time + datetime.timedelta(hours=1),
                       name="empty")
        a.right_merge(empty)
        assert a.data.size == 5

    def test_overlap_raises(self, sample_start_time):
        a = Data1D(data=np.arange(5.0), taxis=np.arange(5.0),
                   start_time=sample_start_time, name="a")
        # a ends at +4s; b starts at +2s -> overlap
        b = Data1D(data=np.arange(3.0), taxis=np.arange(3.0),
                   start_time=sample_start_time + datetime.timedelta(seconds=2),
                   name="b")
        with pytest.raises(ValueError):
            a.right_merge(b)

    def test_non_data1d_raises_typeerror(self, linear_data):
        with pytest.raises(TypeError):
            linear_data.right_merge("not a Data1D")

    def test_missing_start_time_raises(self, sample_start_time):
        a = Data1D(data=np.arange(5.0), taxis=np.arange(5.0),
                   start_time=sample_start_time)
        b = Data1D(data=np.arange(3.0), taxis=np.arange(3.0))  # no start_time
        with pytest.raises(ValueError):
            a.right_merge(b)


# ---------------------------------------------------------------------------
# remove_abnormal_data
# ---------------------------------------------------------------------------
class TestRemoveAbnormalData:
    def test_mean_method(self, sample_start_time):
        d = Data1D(data=np.array([0.0, 1.0, 1000.0, 3.0, 4.0]),
                   taxis=np.arange(5.0), start_time=sample_start_time)
        d.remove_abnormal_data(threshold=10.0, method="mean")
        assert d.data[2] == 2.0  # (1 + 3) / 2

    def test_interp_method(self, sample_start_time):
        d = Data1D(data=np.array([0.0, 1.0, 1000.0, 3.0, 4.0]),
                   taxis=np.arange(5.0), start_time=sample_start_time)
        d.remove_abnormal_data(threshold=10.0, method="interp")
        assert_allclose(d.data, np.array([0.0, 1.0, 2.0, 3.0, 4.0]))

    def test_nan_method(self, sample_start_time):
        d = Data1D(data=np.array([0.0, 1.0, 1000.0, 3.0, 4.0]),
                   taxis=np.arange(5.0), start_time=sample_start_time)
        d.remove_abnormal_data(threshold=10.0, method="nan")
        assert np.isnan(d.data[2])
        assert d.data[1] == 1.0 and d.data[3] == 3.0

    def test_too_few_points_noop(self, sample_start_time):
        d = Data1D(data=np.array([1.0, 1000.0]), taxis=np.array([0.0, 1.0]),
                   start_time=sample_start_time)
        d.remove_abnormal_data(threshold=10.0)
        assert_array_equal(d.data, np.array([1.0, 1000.0]))

    def test_no_abnormal_points_unchanged(self, sample_start_time):
        orig = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        d = Data1D(data=orig.copy(), taxis=np.arange(5.0),
                   start_time=sample_start_time)
        d.remove_abnormal_data(threshold=10.0, method="mean")
        assert_array_equal(d.data, orig)

    def test_empty_data_raises(self, sample_start_time):
        d = Data1D(data=np.array([]), taxis=np.array([]),
                   start_time=sample_start_time)
        with pytest.raises(ValueError):
            d.remove_abnormal_data()


# ---------------------------------------------------------------------------
# lpfilter
# ---------------------------------------------------------------------------
class TestLpFilter:
    def test_lpfilter_preserves_length_and_attenuates_hf(self, sample_start_time):
        taxis = np.linspace(0, 10, 200)
        data = np.sin(2 * np.pi * 1 * taxis) + 0.5 * np.sin(2 * np.pi * 20 * taxis)
        d = Data1D(data=data.copy(), taxis=taxis.copy(),
                   start_time=sample_start_time)
        peak_before = float(np.max(np.abs(d.data)))
        d.lpfilter(freqcut=3.0, order=2)
        assert d.data.size == 200
        # high-frequency component removed -> peak amplitude reduced toward ~1
        assert float(np.max(np.abs(d.data))) < peak_before

    def test_lpfilter_empty_raises(self, sample_start_time):
        d = Data1D(data=np.array([]), taxis=np.array([]),
                   start_time=sample_start_time)
        with pytest.raises(ValueError):
            d.lpfilter(freqcut=1.0)

    def test_lpfilter_single_point_noop(self, sample_start_time):
        d = Data1D(data=np.array([1.0]), taxis=np.array([0.0]),
                   start_time=sample_start_time)
        d.lpfilter(freqcut=1.0)  # size < 2 -> warning + return
        assert_array_equal(d.data, np.array([1.0]))


# ---------------------------------------------------------------------------
# interpolate
# ---------------------------------------------------------------------------
class TestInterpolate:
    def test_nan_fill_outside_range(self, sample_start_time):
        d = Data1D(data=np.array([0.0, 10.0, 20.0]),
                   taxis=np.array([0.0, 1.0, 2.0]),
                   start_time=sample_start_time)
        d.interpolate(np.array([-1.0, 0.5, 1.5, 3.0]))
        assert np.isnan(d.data[0])
        assert d.data[1] == 5.0
        assert d.data[2] == 15.0
        assert np.isnan(d.data[3])

    def test_extrapolate(self, sample_start_time):
        d = Data1D(data=np.array([0.0, 10.0, 20.0]),
                   taxis=np.array([0.0, 1.0, 2.0]),
                   start_time=sample_start_time)
        d.interpolate(np.array([-1.0, 0.5, 3.0]),
                      fill_value_left="extrapolate",
                      fill_value_right="extrapolate")
        assert_allclose(d.data, np.array([-10.0, 5.0, 30.0]))

    def test_new_start_time_offsets_lookup(self, sample_start_time):
        d = Data1D(data=np.array([0.0, 10.0, 20.0]),
                   taxis=np.array([0.0, 1.0, 2.0]),
                   start_time=sample_start_time)
        new_start = sample_start_time + datetime.timedelta(seconds=1)
        # new_taxis [0,1] relative to new_start == original [1,2]
        d.interpolate(np.array([0.0, 1.0]), new_start_time=new_start)
        assert_allclose(d.data, np.array([10.0, 20.0]))
        assert d.start_time == new_start

    def test_empty_new_taxis_raises(self, sample_start_time):
        d = Data1D(data=np.array([0.0, 1.0]), taxis=np.array([0.0, 1.0]),
                   start_time=sample_start_time)
        with pytest.raises(ValueError):
            d.interpolate(np.array([]))

    def test_non_monotonic_new_taxis_raises(self, sample_start_time):
        d = Data1D(data=np.array([0.0, 1.0, 2.0]),
                   taxis=np.array([0.0, 1.0, 2.0]),
                   start_time=sample_start_time)
        with pytest.raises(ValueError):
            d.interpolate(np.array([2.0, 1.0, 0.0]))

    def test_empty_source_raises(self, sample_start_time):
        d = Data1D(data=np.array([]), taxis=np.array([]),
                   start_time=sample_start_time)
        with pytest.raises(ValueError):
            d.interpolate(np.array([0.0, 1.0]))


# ---------------------------------------------------------------------------
# savez (roundtrip via load_npz)
# ---------------------------------------------------------------------------
class TestSavez:
    def test_savez_roundtrip(self, linear_data, tmp_path):
        fn = tmp_path / "out.npz"
        linear_data.savez(str(fn))
        assert fn.exists()

        reloaded = Data1D()
        reloaded.load_npz(str(fn))
        assert_array_equal(reloaded.data, linear_data.data)
        assert_array_equal(reloaded.taxis, linear_data.taxis)
        assert reloaded.start_time == linear_data.start_time

    def test_savez_appends_npz_extension(self, linear_data, tmp_path):
        base = tmp_path / "noext"
        linear_data.savez(str(base))
        assert (tmp_path / "noext.npz").exists()

    def test_savez_empty_raises(self, sample_start_time, tmp_path):
        d = Data1D(data=np.array([]), taxis=np.array([]),
                   start_time=sample_start_time)
        with pytest.raises(ValueError):
            d.savez(str(tmp_path / "x.npz"))


# ---------------------------------------------------------------------------
# adaptive_downsample
# ---------------------------------------------------------------------------
class TestAdaptiveDownsample:
    def test_preserves_peak(self, sample_start_time):
        taxis = np.linspace(0, 10, 21)
        data = np.zeros(21)
        data[10] = 5.0  # a single sharp peak in the middle
        d = Data1D(data=data.copy(), taxis=taxis.copy(),
                   start_time=sample_start_time)
        d.adaptive_downsample(3)
        assert_allclose(d.taxis, np.array([0.0, 5.0, 10.0]))
        assert_allclose(d.data, np.array([0.0, 5.0, 0.0]))

    def test_n_points_ge_size_noop(self, sample_start_time):
        d = Data1D(data=np.arange(5.0), taxis=np.arange(5.0),
                   start_time=sample_start_time)
        d.adaptive_downsample(10)  # >= current size -> warning + return
        assert d.data.size == 5

    def test_n_points_too_small_raises(self, sample_start_time):
        d = Data1D(data=np.arange(10.0), taxis=np.arange(10.0),
                   start_time=sample_start_time)
        with pytest.raises(ValueError):
            d.adaptive_downsample(2)

    def test_n_points_float_raises(self, sample_start_time):
        d = Data1D(data=np.arange(10.0), taxis=np.arange(10.0),
                   start_time=sample_start_time)
        with pytest.raises(ValueError):
            d.adaptive_downsample(3.5)

    def test_empty_raises(self, sample_start_time):
        d = Data1D(data=np.array([]), taxis=np.array([]),
                   start_time=sample_start_time)
        with pytest.raises(ValueError):
            d.adaptive_downsample(3)
