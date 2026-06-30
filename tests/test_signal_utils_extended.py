# Characterization (golden-master) tests for fiberis.utils.signal_utils.
#
# These pin the CURRENT observable behavior of the signal-processing helpers
# that are NOT already covered by tests/test_signal_utils.py. Golden values
# were produced by RUNNING the code; they should survive a pure refactor.
#
# All randomness uses a FIXED seed so the golden numbers are reproducible.

import datetime

import numpy as np
import numpy.testing as npt
import pytest

from fiberis.utils import signal_utils as su


# ---------------------------------------------------------------------------
# Deterministic fixtures
# ---------------------------------------------------------------------------

FS = 100.0
DT = 1.0 / FS


@pytest.fixture
def noisy_sine():
    """5 Hz sine + Gaussian noise, fixed seed -> reproducible golden values."""
    np.random.seed(42)
    t = np.arange(0, 2, DT)
    clean = np.sin(2 * np.pi * 5 * t)
    noise = np.random.randn(len(t)) * 0.5
    return t, clean, clean + noise


# ---------------------------------------------------------------------------
# Butterworth coefficient designers
# ---------------------------------------------------------------------------

def test_butter_bandpass_coeffs():
    b, a = su.butter_bandpass(5.0, 15.0, 100.0, order=2)
    npt.assert_allclose(b, [0.067455, 0.0, -0.134911, 0.0, 0.067455], atol=1e-6)
    npt.assert_allclose(a, [1.0, -2.673579, 2.992362, -1.674577, 0.412802], atol=1e-6)


def test_butter_lppass_coeffs():
    b, a = su.butter_lppass(10.0, 100.0, order=2)
    npt.assert_allclose(b, [0.067455, 0.134911, 0.067455], atol=1e-6)
    npt.assert_allclose(a, [1.0, -1.142981, 0.412802], atol=1e-6)


def test_butter_hppass_coeffs():
    b, a = su.butter_hppass(10.0, 100.0, order=2)
    npt.assert_allclose(b, [0.638946, -1.277891, 0.638946], atol=1e-6)
    npt.assert_allclose(a, [1.0, -1.142981, 0.412802], atol=1e-6)


# ---------------------------------------------------------------------------
# Filters applied to data
# ---------------------------------------------------------------------------

def test_bpfilter_golden(noisy_sine):
    _, _, sig = noisy_sine
    y = su.bpfilter(sig, DT, 3.0, 7.0)
    assert y.shape == sig.shape
    npt.assert_allclose(np.sum(y), 2.900411, atol=1e-5)
    npt.assert_allclose(
        y[:5], [0.11353, 0.491637, 0.83192, 1.10011, 1.268529], atol=1e-5
    )


def test_lpfilter_golden(noisy_sine):
    _, _, sig = noisy_sine
    y = su.lpfilter(sig, DT, 10.0)
    npt.assert_allclose(np.sum(y), -4.321422, atol=1e-5)
    npt.assert_allclose(
        y[:5], [0.24754, 0.548468, 0.825142, 1.04228, 1.176369], atol=1e-5
    )


def test_hpfilter_golden(noisy_sine):
    _, _, sig = noisy_sine
    y = su.hpfilter(sig, DT, 10.0)
    npt.assert_allclose(np.sum(y), 0.23936, atol=1e-5)
    npt.assert_allclose(
        y[:5], [0.000817, -0.308583, 0.086487, 0.528252, -0.342389], atol=1e-5
    )


# ---------------------------------------------------------------------------
# Amplitude spectrum
# ---------------------------------------------------------------------------

def test_amp_spectrum_peak_and_length():
    t = np.arange(0, 2, DT)
    clean = np.sin(2 * np.pi * 5 * t)
    freqs, amps = su.amp_spectrum(clean, DT)
    # Only positive (>=0) frequencies returned.
    assert len(freqs) == 100
    assert np.all(freqs >= 0)
    peak_freq = freqs[np.argmax(amps)]
    npt.assert_allclose(peak_freq, 5.0, atol=1e-6)
    npt.assert_allclose(np.max(amps), 100.0, atol=1e-2)


def test_amp_spectrum_ortho_norm_scales():
    t = np.arange(0, 2, DT)
    clean = np.sin(2 * np.pi * 5 * t)
    _, amps_plain = su.amp_spectrum(clean, DT)
    _, amps_ortho = su.amp_spectrum(clean, DT, norm="ortho")
    # ortho norm divides by sqrt(N); N=200 here.
    npt.assert_allclose(amps_ortho, amps_plain / np.sqrt(200), atol=1e-9)


# ---------------------------------------------------------------------------
# rms
# ---------------------------------------------------------------------------

def test_rms_1d():
    npt.assert_allclose(su.rms(np.array([3.0, 4.0])), 3.5355339059327378, atol=1e-12)


def test_rms_axes():
    m = np.array([[1.0, 2.0], [3.0, 4.0]])
    npt.assert_allclose(su.rms(m, axis=0), [2.236068, 3.162278], atol=1e-6)
    npt.assert_allclose(su.rms(m, axis=1), [1.581139, 3.535534], atol=1e-6)


# ---------------------------------------------------------------------------
# phase_wrap
# ---------------------------------------------------------------------------

def test_phase_wrap_golden():
    out = su.phase_wrap(np.array([0, np.pi, 2 * np.pi, 3 * np.pi, -np.pi]))
    # np.angle wraps to (-pi, pi]; 2*pi -> 0, 3*pi -> pi, -pi -> -pi.
    npt.assert_allclose(out, [0.0, np.pi, 0.0, np.pi, -np.pi], atol=1e-6)


# ---------------------------------------------------------------------------
# matdatenum_to_pydatetime
# ---------------------------------------------------------------------------

def test_matdatenum_epoch():
    assert su.matdatenum_to_pydatetime(719529) == datetime.datetime(1970, 1, 1)


def test_matdatenum_recent():
    assert su.matdatenum_to_pydatetime(738886) == datetime.datetime(2022, 12, 31)


# ---------------------------------------------------------------------------
# correlation_coefficient
# ---------------------------------------------------------------------------

def test_correlation_perfect():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    n = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    npt.assert_allclose(su.correlation_coefficient(a, n), 1.0, atol=1e-12)


def test_correlation_mismatched_length_returns_none():
    assert su.correlation_coefficient(np.array([1.0, 2.0]), np.array([1.0])) is None


def test_correlation_empty_returns_none():
    assert su.correlation_coefficient(np.array([]), np.array([])) is None


def test_correlation_flat_equal_returns_one():
    # Both constant AND equal -> returns 1.0 (zero-variance special case).
    a = np.array([2.0, 2.0, 2.0])
    assert su.correlation_coefficient(a, a) == 1.0


def test_correlation_flat_one_side_returns_zero():
    # One side constant -> denominator zero, numerator zero -> returns 0.0.
    a = np.array([2.0, 2.0, 2.0])
    n = np.array([1.0, 2.0, 3.0])
    assert su.correlation_coefficient(a, n) == 0.0


def test_correlation_random_golden():
    np.random.seed(1)
    a = np.random.randn(50)
    n = np.random.randn(50)
    npt.assert_allclose(su.correlation_coefficient(a, n), -0.02255187, atol=1e-7)


# ---------------------------------------------------------------------------
# Interpolation matrices
# ---------------------------------------------------------------------------

def test_get_interp_mat_linear():
    m = su.get_interp_mat(3, 5, kind="linear")
    expected = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.0, 0.0, 1.0],
        ]
    )
    npt.assert_allclose(m, expected, atol=1e-9)


def test_get_interp_mat_quadratic_rows_sum_to_one():
    m = su.get_interp_mat(3, 5, kind="quadratic")
    npt.assert_allclose(m.sum(axis=1), np.ones(5), atol=1e-9)


def test_get_interp_mat_anchorx_linear_endpoints():
    x = np.linspace(0, 10, 6)
    ax = np.array([0.0, 5.0, 10.0])
    m = su.get_interp_mat_anchorx(x, ax, kind="linear")
    assert m.shape == (6, 3)
    npt.assert_allclose(m[0], [1.0, 0.0, 0.0], atol=1e-9)
    npt.assert_allclose(m[-1], [0.0, 0.0, 1.0], atol=1e-9)


# ---------------------------------------------------------------------------
# get_smooth_curve
# ---------------------------------------------------------------------------

def test_get_smooth_curve_linear_golden():
    np.random.seed(7)
    x0 = np.linspace(0, 10, 50)
    data = 2 * x0 + 1 + np.random.randn(50) * 0.1
    anchor_x = np.linspace(0, 10, 4)
    sm, solved = su.get_smooth_curve(x0, anchor_x, data, kind="linear")
    assert len(sm) == 50
    npt.assert_allclose(sm[:3], [1.021991, 1.426864, 1.831736], atol=1e-5)
    npt.assert_allclose(
        solved, [1.021991, 7.63491, 14.300678, 20.963986], atol=1e-5
    )


# ---------------------------------------------------------------------------
# running_average
# ---------------------------------------------------------------------------

def test_running_average_odd_window():
    d = np.arange(10, dtype=float)
    ra = su.running_average(d, 3)
    npt.assert_allclose(
        ra, [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.5], atol=1e-9
    )


def test_running_average_even_window():
    d = np.arange(10, dtype=float)
    ra = su.running_average(d, 4)
    npt.assert_allclose(
        ra, [1.0, 1.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.0], atol=1e-9
    )


def test_running_average_nonpositive_window_returns_copy():
    d = np.arange(10, dtype=float)
    out = su.running_average(d, 0)
    npt.assert_array_equal(out, d)
    assert out is not d  # returns a copy


def test_running_average_window_larger_than_data():
    d = np.array([1.0, 2.0, 3.0])
    out = su.running_average(d, 5)
    npt.assert_allclose(out, [2.0, 2.0, 2.0], atol=1e-9)


# ---------------------------------------------------------------------------
# xcor_match
# ---------------------------------------------------------------------------

def test_xcor_match_shifted_impulse():
    a = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    npt.assert_allclose(su.xcor_match(a, b), 1.0, atol=1e-12)


def test_xcor_match_identical_zero_lag():
    a = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    npt.assert_allclose(su.xcor_match(a, a), 0.0, atol=1e-12)


def test_xcor_match_empty_returns_nan():
    assert np.isnan(su.xcor_match(np.array([]), np.array([])))


def test_xcor_match_unequal_length_returns_nan():
    assert np.isnan(su.xcor_match(np.array([1.0, 2.0]), np.array([1.0])))


def test_xcor_match_flat_returns_nan():
    assert np.isnan(su.xcor_match(np.zeros(5), np.zeros(5)))


# ---------------------------------------------------------------------------
# datetime_interp
# ---------------------------------------------------------------------------

def test_datetime_interp_basic():
    t0 = datetime.datetime(2023, 1, 1)
    timex0 = [t0 + datetime.timedelta(seconds=s) for s in (0, 10, 20)]
    y0 = np.array([0.0, 10.0, 20.0])
    timex = [t0 + datetime.timedelta(seconds=s) for s in (5, 15)]
    npt.assert_allclose(su.datetime_interp(timex, timex0, y0), [5.0, 15.0], atol=1e-9)


def test_datetime_interp_empty():
    out = su.datetime_interp([], [], np.array([]))
    assert out.size == 0


def test_datetime_interp_single_reference():
    t0 = datetime.datetime(2023, 1, 1)
    timex = [t0 + datetime.timedelta(seconds=s) for s in (5, 15)]
    out = su.datetime_interp(timex, [t0], np.array([42.0]))
    npt.assert_allclose(out, [42.0, 42.0], atol=1e-12)


# ---------------------------------------------------------------------------
# fetch_timestamp_fast
# ---------------------------------------------------------------------------

def test_fetch_timestamp_fast_basic():
    strs = [
        "2023-01-01 00:00:00",
        "2023-01-01 00:00:01",
        "2023-01-01 00:00:02",
        "2023-01-01 00:00:03",
    ]
    ts, t = su.fetch_timestamp_fast(strs, downsampling=1)
    assert ts[0] == datetime.datetime(2023, 1, 1, 0, 0, 0)
    npt.assert_allclose(t, [0.0, 1.0, 2.0, 3.0], atol=1e-9)


def test_fetch_timestamp_fast_single():
    ts, t = su.fetch_timestamp_fast(["2023-01-01 00:00:00"], downsampling=1)
    assert ts[0] == datetime.datetime(2023, 1, 1, 0, 0, 0)
    npt.assert_allclose(t, [0.0], atol=1e-12)


def test_fetch_timestamp_fast_empty():
    ts, t = su.fetch_timestamp_fast([])
    assert ts == []
    assert t.size == 0


# ---------------------------------------------------------------------------
# timeshift_xcor
# ---------------------------------------------------------------------------

def test_timeshift_xcor_detects_shift():
    np.random.seed(11)
    N = 100
    base = np.sin(2 * np.pi * np.arange(N) / 20)
    data2 = base
    data1 = np.roll(base, 3)
    ts, shifted = su.timeshift_xcor(data1, data2, winsize=20, step=2)
    assert ts.shape == (N,)
    assert shifted.shape == (N,)
    # NOTE: golden values pin current behavior (sign/convention not asserted).
    npt.assert_allclose(np.nanmean(ts), -2.574996, atol=1e-5)
    npt.assert_allclose(
        ts[:5],
        [-2.999412, -2.99892, -2.998435, -2.998281, -2.999032],
        atol=1e-5,
    )
    npt.assert_allclose(np.sum(shifted), 0.222961, atol=1e-5)


def test_timeshift_xcor_empty():
    ts, sd = su.timeshift_xcor(np.array([]), np.array([]), winsize=5)
    assert ts.size == 0
    assert sd.size == 0


def test_timeshift_xcor_winsize_larger_than_data():
    d = np.arange(10.0)
    ts, sd = su.timeshift_xcor(d, d, winsize=20)
    assert np.all(np.isnan(ts))
    npt.assert_array_equal(sd, d)


def test_timeshift_xcor_winsize_zero():
    d = np.arange(10.0)
    ts, sd = su.timeshift_xcor(d, d, winsize=0)
    assert np.all(np.isnan(ts))
