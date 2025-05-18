# Dr.Jin's functions. I did some refactoring and added some tweaks.
# Shenyao, shenyaojin@mines.edu
import sys
import datetime
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr
from dateutil.parser import parse

def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Designs a Butterworth bandpass filter.

    Args:
        lowcut: Lower cutoff frequency.
        highcut: Upper cutoff frequency.
        fs: Sampling frequency.
        order: Order of the filter.

    Returns:
        Numerator (b) and denominator (a) polynomials of the IIR filter.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_lppass(freqcut: float, fs: float, order: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Designs a Butterworth lowpass filter.

    Args:
        freqcut: Cutoff frequency.
        fs: Sampling frequency.
        order: Order of the filter.

    Returns:
        Numerator (b) and denominator (a) polynomials of the IIR filter.
    """
    nyq = 0.5 * fs
    w = freqcut / nyq
    b, a = butter(order, w, btype='low')
    return b, a


def butter_hppass(freqcut: float, fs: float, order: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Designs a Butterworth highpass filter.

    Args:
        freqcut: Cutoff frequency.
        fs: Sampling frequency.
        order: Order of the filter.

    Returns:
        Numerator (b) and denominator (a) polynomials of the IIR filter.
    """
    nyq = 0.5 * fs
    w = freqcut / nyq
    b, a = butter(order, w, btype='high')
    return b, a


def bpfilter(data: np.ndarray, dt: float, lowcut: float, highcut: float, order: int = 2, axis: int = -1) -> np.ndarray:
    """
    Applies a bandpass filter to data.

    Args:
        data: Input data array.
        dt: Sampling interval.
        lowcut: Lower cutoff frequency.
        highcut: Upper cutoff frequency.
        order: Order of the filter.
        axis: Axis along which to filter.

    Returns:
        Filtered data.
    """
    fs = 1 / dt
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=axis)
    return y


def lpfilter(data: np.ndarray, dt: float, freqcut: float, order: int = 2, plotSpectrum: bool = False,
             axis: int = -1) -> np.ndarray:
    """
    Applies a lowpass filter to data.

    Args:
        data: Input data array.
        dt: Sampling interval.
        freqcut: Cutoff frequency.
        order: Order of the filter.
        plotSpectrum: (Not used in current implementation) Whether to plot the spectrum.
        axis: Axis along which to filter.

    Returns:
        Filtered data.
    """
    fs = 1 / dt
    b, a = butter_lppass(freqcut, fs, order=order)
    y = filtfilt(b, a, data, axis=axis)
    # plotSpectrum functionality would need to be implemented if desired
    return y


def hpfilter(data: np.ndarray, dt: float, freqcut: float, order: int = 2, axis: int = -1) -> np.ndarray:
    """
    Applies a highpass filter to data.

    Args:
        data: Input data array.
        dt: Sampling interval.
        freqcut: Cutoff frequency.
        order: Order of the filter.
        axis: Axis along which to filter.

    Returns:
        Filtered data.
    """
    fs = 1 / dt
    b, a = butter_hppass(freqcut, fs, order=order)
    y = filtfilt(b, a, data, axis=axis)
    return y


def amp_spectrum(data: np.ndarray, dt: float, norm: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the amplitude spectrum of a 1D signal.

    Args:
        data: Input 1D data array.
        dt: Sampling interval.
        norm: Normalization mode for FFT (e.g., "ortho").

    Returns:
        Frequencies and corresponding amplitude spectrum.
    """
    # data.flatten() creates a copy, which is often safer.
    # data.ravel() might return a view, potentially faster if no copy is needed
    # but FFT might copy anyway. Sticking to flatten for explicitness.
    flat_data = data.flatten()
    N = len(flat_data)
    freqs = np.fft.fftfreq(N, dt)
    asp = np.abs(np.fft.fft(flat_data, norm=norm))

    idx = np.argsort(freqs)
    freqs_sorted = freqs[idx]
    asp_sorted = asp[idx]

    # Return only positive frequencies
    positive_freq_indices = freqs_sorted >= 0
    return freqs_sorted[positive_freq_indices], asp_sorted[positive_freq_indices]


def samediff(data: np.ndarray) -> np.ndarray:
    """
    Computes the difference between adjacent elements, appending the last difference
    to maintain the original array's length.

    Args:
        data: Input array.

    Returns:
        Array of differences with the same length as input.
    """
    if data.size == 0:
        return np.array([])
    if data.size == 1:
        return np.array([0.])  # Or handle as an error, or return data itself
    y = np.diff(data)
    y = np.append(y, y[-1])
    return y


def fillnan(data: np.ndarray) -> np.ndarray:
    """
    Fills NaN values in a 1D array using linear interpolation.

    Args:
        data: Input 1D array.

    Returns:
        Array with NaNs filled.
    """
    if not np.any(np.isnan(data)):
        return data
    if np.all(np.isnan(data)):  # All NaNs, nothing to interpolate from
        return data

    ind = ~np.isnan(data)
    x = np.arange(len(data))  # More direct than np.array(range(len(data)))

    # Edge case: if first or last values are NaN, interp might extrapolate
    # or one might prefer to fill with first/last valid value (ffill/bfill)
    # For now, using standard interp behavior.
    y = np.interp(x, x[ind], data[ind])
    return y


def timediff(ts1: datetime.datetime, ts2: datetime.datetime) -> float:
    """
    Calculates the time difference between two datetime objects in seconds.

    Args:
        ts1: First datetime object.
        ts2: Second datetime object.

    Returns:
        Difference in seconds (ts1 - ts2).
    """
    tdiff = ts1 - ts2
    return tdiff.total_seconds()  # Simpler and more direct


def get_interp_mat(anchor_N: int, N: int, kind: str = 'quadratic') -> np.ndarray:
    """
    Generates an interpolation matrix.
    Rows correspond to the interpolated points, columns to anchor points.

    Args:
        anchor_N: Number of anchor points.
        N: Number of points to interpolate.
        kind: Type of interpolation (e.g., 'linear', 'quadratic', 'cubic').

    Returns:
        Interpolation matrix (N x anchor_N).
    """
    x = np.arange(N)
    anchor_x = np.linspace(x[0], x[-1], anchor_N)
    interp_mat = np.zeros((N, anchor_N))

    # This loop creates an interpolator for each basis vector
    for i in range(anchor_N):
        test_y = np.zeros(anchor_N, )
        test_y[i] = 1
        # Handle case where anchor_N is too small for certain kinds of interpolation
        if kind in ['quadratic', 'cubic'] and anchor_N < {'quadratic': 3, 'cubic': 4}.get(kind, 1):
            # Fallback to linear if not enough points for higher order
            current_kind = 'linear'
            if anchor_N < 2:  # Need at least 2 points for linear
                current_kind = 'previous'  # or 'next', or handle error
                if anchor_N < 1:
                    interp_mat[:, i] = 0 if anchor_N == 0 else 1  # or raise error
                    continue
        else:
            current_kind = kind

        f_interp = interp1d(anchor_x, test_y, kind=current_kind, bounds_error=False, fill_value="extrapolate")
        interp_mat[:, i] = f_interp(x)
    return interp_mat


def get_interp_mat_anchorx(x: np.ndarray, anchor_x: np.ndarray, kind: str = 'quadratic') -> np.ndarray:
    """
    Generates an interpolation matrix given specific x and anchor_x coordinates.

    Args:
        x: X-coordinates of points to interpolate.
        anchor_x: X-coordinates of anchor points.
        kind: Type of interpolation.

    Returns:
        Interpolation matrix (len(x) x len(anchor_x)).
    """
    anchor_N = len(anchor_x)
    N = len(x)
    interp_mat = np.zeros((N, anchor_N))

    for i in range(anchor_N):
        test_y = np.zeros(anchor_N, )
        test_y[i] = 1

        current_kind = kind
        # Fallback for insufficient points for interpolation kind
        if kind in ['quadratic', 'cubic'] and anchor_N < {'quadratic': 3, 'cubic': 4}.get(kind, 1):
            current_kind = 'linear'
            if anchor_N < 2:
                current_kind = 'previous'
                if anchor_N < 1:
                    interp_mat[:, i] = 0 if anchor_N == 0 else 1
                    continue

        f_interp = interp1d(anchor_x, test_y, kind=current_kind, bounds_error=False, fill_value="extrapolate")
        interp_mat[:, i] = f_interp(x)
    return interp_mat


def get_smooth_curve(x0: np.ndarray, anchor_x: np.ndarray, data: np.ndarray,
                     kind: str = 'quadratic', errstd: float = 3, iterN: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Smooths a curve using interpolation and iterative outlier rejection.

    Args:
        x0: Original x-coordinates of the data.
        anchor_x: X-coordinates of anchor points for smoothing.
        data: Original y-data.
        kind: Type of interpolation for smoothing.
        errstd: Standard deviation multiplier for outlier rejection.
        iterN: Number of iterations for outlier rejection.

    Returns:
        Smoothed data array and the solved anchor point values (x).
    """
    iterdata = data.copy()
    iterx = x0.copy()

    solved_x = None  # Keep track of the last solved x

    for _ in range(iterN + 1):
        if len(iterdata) < len(anchor_x) or len(iterdata) == 0:  # Not enough points to solve
            break
        interp_mat = get_interp_mat_anchorx(iterx, anchor_x, kind=kind)
        # Using rcond=None to use default and suppress future warnings
        current_solved_x = np.linalg.lstsq(interp_mat, iterdata, rcond=None)[0]

        if solved_x is None:  # Store the first solution
            solved_x = current_solved_x

        err_sq = (iterdata - np.dot(interp_mat, current_solved_x)) ** 2  # Square of errors

        if len(err_sq) == 0:  # No data left
            break

        std_err = np.std(err_sq)
        if std_err == 0:  # All errors are same, no basis for outlier removal
            solved_x = current_solved_x  # Update solved_x with current
            break

        goodind = err_sq < std_err * errstd  # Threshold on squared error (variance) or abs error? Original was on squared.
        # goodind = np.abs(iterdata - np.dot(interp_mat, current_solved_x)) < np.std(iterdata - np.dot(interp_mat, current_solved_x)) * errstd # Threshold on error itself

        if not np.any(goodind) or np.sum(goodind) < len(anchor_x):  # Not enough good points left
            break  # Use the previous iteration's solution

        iterdata = iterdata[goodind]
        iterx = iterx[goodind]
        solved_x = current_solved_x  # Update solved_x

    # If loop didn't run or broke early, use initial data or handle error
    if solved_x is None:
        # Fallback: if no iterations were successful, do a single pass without outlier rejection
        interp_mat_final = get_interp_mat_anchorx(x0, anchor_x, kind=kind)
        if x0.shape[0] >= anchor_x.shape[0] and x0.shape[0] > 0:  # Check if lstsq is possible
            solved_x = np.linalg.lstsq(interp_mat_final, data, rcond=None)[0]
            smdata = np.dot(interp_mat_final, solved_x)
            return smdata, solved_x
        else:  # Cannot compute, return original data or NaNs
            return data.copy(), np.full(len(anchor_x), np.nan)

    interp_mat_final = get_interp_mat_anchorx(x0, anchor_x, kind=kind)
    smdata = np.dot(interp_mat_final, solved_x)
    return smdata, solved_x


def rms(a: np.ndarray, axis: int | None = None) -> float | np.ndarray:
    """
    Calculates the root mean square (RMS) of an array.

    Args:
        a: Input array.
        axis: Axis along which to compute RMS. If None, computes over the entire array.

    Returns:
        RMS value(s).
    """
    return np.sqrt(np.mean(a ** 2, axis=axis))


def matdatenum_to_pydatetime(matlab_datenum: float) -> datetime.datetime:
    """
    Converts a MATLAB datenum to a Python datetime object.

    Args:
        matlab_datenum: MATLAB serial date number.

    Returns:
        Python datetime object.
    """
    # MATLAB's epoch is January 0, 0000. Python's is January 1, 0001.
    # MATLAB's datenum for 0000-01-01 is 1.
    # Ordinal for 0001-01-01 is 1.
    # Python's datetime.fromordinal(1) is 0001-01-01.
    # MATLAB datenum 1 -> Python datetime.fromordinal(1) + timedelta(days=1%1) - timedelta(days=366)
    # Let's test with a known value: 719529 (Jan 1, 1970)
    # python_datetime = datetime.datetime.fromordinal(int(matlab_datenum)) + \
    #                   timedelta(days=matlab_datenum % 1) - \
    #                   timedelta(days=366) # This is the common formula
    # A more direct way:
    # MATLAB epoch starts on datenum 0 = 00-Jan-0000
    # Python ordinal for 1-Jan-0001 is 1.
    # Difference is 366 days (day 1 in MATLAB is 0000-01-01, day 1 in Python is 0001-01-01)
    # Or, using a known reference point like MATLAB's datenum for 1970-01-01 (719529)
    # and Python's datetime(1970,1,1).
    days = matlab_datenum - 366.0  # Adjust for Python's ordinal base (1-Jan-0001)
    # from MATLAB's (0-Jan-0000 interpreted as 1-Jan-0000 for date calcs)
    # No, the original offset is from a different interpretation.
    # Standard formula:
    python_datetime = datetime.datetime.fromordinal(int(matlab_datenum + 366)) + \
                      timedelta(days=matlab_datenum % 1) - \
                      timedelta(days=366)
    # The original formula was:
    # python_datetime = datetime.datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum % 1) - timedelta(days=366)
    # This works because datetime.fromordinal(1) is 0001-01-01.
    # And MATLAB's datenum for 0001-01-01 is 367.
    # So, if matlab_datenum = 367, int(367) - 366 = 1. fromordinal(1) is 0001-01-01. Correct.
    return python_datetime


def rfft_xcorr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes cross-correlation using real FFT.
    The output is structured to be used by xcor_match.

    Args:
        x: First signal.
        y: Second signal.

    Returns:
        Cross-correlation result array.
        The length is len(x) + len(y) - 1.
        The array is constructed by concatenating:
        - correlation values for non-negative lags (y shifted right w.r.t x, up to len(x)-1 shifts)
        - correlation values for negative lags (y shifted left w.r.t x)
    """
    M = len(x) + len(y) - 1  # Length of full linear cross-correlation
    # Pad to next power of 2 for FFT efficiency
    N = 2 ** int(np.ceil(np.log2(M)))

    X = np.fft.rfft(x, N)
    Y = np.fft.rfft(y, N)

    # Conjugate Y for cross-correlation
    cxy_full = np.fft.irfft(X * np.conj(Y), N)

    # The hstack below reconstructs a specific ordering of correlation lags.
    # It takes the first len(x) points from cxy_full (representing certain lags)
    # and appends points from the end of cxy_full (representing other lags).
    # This specific arrangement is expected by xcor_match.
    # cxy_full[0] corresponds to zero lag if x and y are perfectly aligned at their starts in circular sense.
    # For linear correlation, interpretation of lags from raw fft output needs care.
    # This function's hstack produces an array of length M.
    # The first len(x) elements correspond to positive lags (0 to len(x)-1).
    # The remaining len(y)-1 elements correspond to negative lags (-(len(y)-1) to -1),
    # but their values in the concatenated array are cxy_full[N - len(y) + 1:]
    # This is a custom arrangement.
    # A standard 'full' correlation would be cxy_full[:M], and lags would typically be
    # interpreted from -(len(y)-1) to len(x)-1.

    # Part 1: Corresponds to lags where y is shifted from 0 to len(x)-1 relative to x's start
    part1 = cxy_full[:len(x)]
    # Part 2: Corresponds to lags where y is shifted from -(len(y)-1) to -1 relative to x's start
    # These are taken from the end of the circular correlation result.
    part2 = cxy_full[N - len(y) + 1:] if len(y) > 1 else np.array([])

    cxy = np.hstack((part1, part2))
    return cxy


def xcor_match(a: np.ndarray, b: np.ndarray, threshold: float = 0.3) -> float:
    """
    Finds the optimal lag between two signals 'a' and 'b' using cross-correlation.
    The lag indicates how much 'b' should be shifted to best match 'a'.

    Args:
        a: First signal (reference).
        b: Second signal (to be shifted).
        threshold: Minimum absolute Pearson correlation to proceed with lag calculation.

    Returns:
        Optimal lag (integer), or np.nan if correlation is below threshold
        or if signals are unsuitable.
        A positive lag means 'b' is delayed relative to 'a' (needs to be shifted left).
        A negative lag means 'b' is advanced relative to 'a' (needs to be shifted right).
        The returned lag is the index in 'a' where the peak of 'b' aligns.
    """
    if len(a) == 0 or len(b) == 0 or len(a) != len(b):  # Added checks for empty or unequal length
        # print("Warning: xcor_match received empty or unequal length arrays.")
        return np.nan

    x = a.copy()
    ref = b.copy()

    # Detrend by removing mean
    x -= np.mean(x)
    ref -= np.mean(ref)

    # Avoid division by zero in pearsonr if signals are flat
    if np.all(x == 0) or np.all(ref == 0):
        return np.nan

    try:
        r, _ = pearsonr(x, ref)
    except ValueError:  # Can happen if variance is zero
        return np.nan

    if abs(r) < threshold:
        return np.nan

    cxy = rfft_xcorr(x, ref)  # Uses the custom rfft_xcorr
    if len(cxy) == 0:  # Should not happen if x, ref are not empty
        return np.nan

    index = np.argmax(cxy)

    # Interpretation of 'index' based on rfft_xcorr's output structure:
    # If index < len(x), it's a non-negative lag. The value 'index' itself.
    # This means ref (b) aligns best with a segment of x starting at 'index'.
    # (e.g., index 0 means b aligns with x[0:len(b)]).
    # If index >= len(x), it's a negative lag.
    # The lag value is index - len(cxy).
    # len(cxy) = len(x) + len(ref) - 1. Since len(x)==len(ref), len(cxy) = 2*len(x)-1.
    # Example: len(x)=10. cxy length 19.
    # Positive lags: index 0..9.
    # Negative lags: index 10..18.
    # If index = 10, lag = 10 - 19 = -9.
    # This lag indicates the shift of 'ref' relative to 'x'.
    # A positive 'index' (0 to len(x)-1) directly implies the starting point in 'x'
    # where 'ref' aligns after being shifted 'index' samples to the right.
    # A lag value derived as `index - (len(ref) - 1)` is more standard for 'full' correlation.
    # The current return value:
    #   - `index` if `index < len(x)`: This is the shift of `ref` to the right to align with `x`.
    #                                 The peak of correlation occurs when `ref` aligns with `x[index : index+len(ref)]`.
    #   - `index - len(cxy)` if `index >= len(x)`: This is a negative shift.
    if index < len(x):
        # Positive lag: ref is shifted right by 'index' samples.
        # Or, x is shifted left by 'index' samples to match ref.
        return float(index)
    else:  # Negative lag
        # ref is shifted left.
        return float(index - len(cxy))


def timeshift_xcor(data1: np.ndarray, data2: np.ndarray, winsize: int,
                   step: int = 1, lowf: float = 1 / 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimates time shifts between two signals using windowed cross-correlation
    and applies the shift to data1.

    Args:
        data1: First signal.
        data2: Second signal (reference).
        winsize: Window size for cross-correlation.
        step: Step size for moving the window.
        lowf: Lowpass filter cutoff frequency for smoothing the calculated time shifts (normalized to dt=1).

    Returns:
        ts: Array of time shifts (dt=1).
        shift_data1: data1 shifted according to 'ts'.
    """
    N = len(data1)
    if N == 0 or len(data2) == 0 or N != len(data2):
        # print("Warning: timeshift_xcor received empty or unequal length arrays.")
        return np.full(N, np.nan), data1.copy()  # Or raise error

    if winsize <= 0 or winsize > N:
        # print("Warning: Invalid winsize for timeshift_xcor.")
        return np.full(N, np.nan), data1.copy()

    ori_x = np.arange(N)
    # Ensure cx starts and ends within array bounds considering window size
    # Center of the first window: winsize // 2
    # Center of the last window: N - 1 - (winsize // 2)
    start_cx = winsize // 2
    end_cx = N - (winsize // 2)  # Exclusive end for arange
    if start_cx >= end_cx:  # Window is too large for the data with given step
        return np.full(N, np.nan), data1.copy()

    cx = np.arange(start_cx, end_cx, step)
    if len(cx) == 0:  # No valid window centers
        return np.full(N, np.nan), data1.copy()

    cy = np.zeros(len(cx))  # Stores estimated shifts (lags from xcor_match)

    for i in range(len(cx)):
        center = cx[i]
        winbg = center - (winsize // 2)
        wined = center + (winsize // 2) + (winsize % 2)  # Ensure window is of size 'winsize'

        segment1 = data1[winbg:wined]
        segment2 = data2[winbg:wined]

        # xcor_match returns the offset of segment2 relative to segment1
        # A positive value means segment2 is delayed (aligned with later part of segment1)
        # A negative value means segment2 is advanced
        lag = xcor_match(segment1, segment2)
        cy[i] = lag  # This lag is the shift data2 needs to align with data1 in the window

    # Interpolate shifts to original time axis
    valid_shifts = ~np.isnan(cy)
    if not np.any(valid_shifts):  # No valid shifts found
        return np.full(N, np.nan), data1.copy()

    # If only one valid shift, interp won't work well; replicate it or handle
    if np.sum(valid_shifts) < 2:
        ts_raw = np.full(N, np.nan)
        if np.sum(valid_shifts) == 1:
            ts_raw[:] = cy[valid_shifts][0]  # Fill with the single valid shift
    else:
        ts_raw = np.interp(ori_x, cx[valid_shifts], cy[valid_shifts])

    # Smooth the time shifts
    # dt=1 for lpfilter as shifts are in samples
    ts = lpfilter(ts_raw, dt=1, freqcut=lowf, order=2)

    # Apply shifts: new_positions = original_positions - estimated_shifts
    # If ts[j] is positive, data2[j] is 'ts[j]' samples 'later' than data1[j] in that window.
    # So, data1[j] corresponds to data2[j - ts[j]].
    # We want to find where the value originally at data1[j] should move to.
    # If data1 is our target to modify based on its alignment with data2:
    #   lag = xcor_match(data1_window, data2_window)
    #   A positive 'lag' means data2_window (ref) is found 'lag' samples into data1_window.
    #   So data1_window[lag] aligns with data2_window[0].
    #   This means data1 is "ahead" of data2 by 'lag' samples in that window.
    #   To align data1 with data2, data1 needs to be shifted right by 'lag'.
    #   So, the values from data1 should be read from (ori_x - ts).
    # The original code uses (ori_x + ts). Let's verify this logic.
    # If ts[i] (from xcor_match(data1, data2)) = L > 0, means data2 aligns with data1[L:].
    # So data1 needs to be shifted to the right by L.
    # target_indices_for_lookup = original_indices - L
    # shift_data1[i] = data1[ original_indices[i] - ts[i] ]
    # np.interp(x_new, x_original, y_original)
    # We want values for original_indices. We need to know where to look up in data1.
    # The values for `shift_data1` at `ori_x` should come from `data1` at `ori_x - ts`.
    # tar_x = ori_x - ts  # These are the source indices in original data1
    # So, shift_data1 = np.interp(ori_x, ori_x, data1_values_at_tar_x) is not right.
    # It should be: for each point in `ori_x`, where does its value come from?
    # The new value at `ori_x[i]` is `data1[ori_x[i] - ts[i]]`.
    # So, `shift_data1 = np.interp(ori_x - ts, ori_x, data1)`
    # The original `tar_x = ori_x + ts` implies shifting data1 to the left if ts is positive.
    # If xcor_match(A,B) gives lag L, B is found at A[L:]. To align A to B, A must be shifted right by L.
    # This means the value at new A[i] comes from old A[i-L].
    # So, new_indices = old_indices - L.
    # `shift_data1 = np.interp(ori_x - ts, ori_x, data1)` is the interpretation for A shifted to align with B.
    # If `shift_data1 = np.interp(ori_x + ts, ori_x, data1)` is used, it means data1 is shifted by -ts.
    # Given the context of "timeshift_xcor", ts is likely the shift *for data1*.
    # Let's stick to the original `tar_x = ori_x + ts` and assume `ts` is defined as the shift to apply to `ori_x`
    # to find the corresponding point in the *original* `data1`.
    # So, `shift_data1[i]` gets its value from `data1[ori_x[i] + ts[i]]`.
    # This means if ts[i] is positive, we look *later* in data1. Data1 is shifted left.

    tar_x = ori_x + ts  # These are source indices in data1 for the new timeline ori_x
    shift_data1 = np.interp(tar_x, ori_x, data1)
    return ts, shift_data1


def running_average(data: np.ndarray, N: int) -> np.ndarray:
    """
    Computes a running average with special handling for edges.
    The values at the edges are means of smaller, asymmetric windows.

    Args:
        data: Input 1D array.
        N: Window size for the running average.

    Returns:
        Array with the running average applied.
    """
    if N <= 0:
        # print("Warning: running_average window N must be positive.")
        return data.copy()
    if N > len(data):  # Window larger than data, return mean of all data
        return np.full_like(data, np.mean(data)) if len(data) > 0 else data.copy()

    # Central part using convolution
    outdata = np.convolve(data, np.ones((N,)) / N, mode='same')

    halfN = N // 2  # Integer division

    # Edge handling: recalculate mean for the first halfN points
    # For these points, the window extends to data[0]
    for i in range(halfN):  # Corrected loop range based on typical half-window use
        # Window for point i: data[0 : i + halfN + 1] if N is odd
        # Window for point i: data[0 : i + halfN] if N is even (assymetric)
        # Let's make it simpler: window from 0 to min(len(data), i + halfN + (N%2))
        # Original logic: outdata[i] = np.mean(data[:i + halfN])
        # This means for i=0, mean(data[:halfN]). For i=halfN, mean(data[:halfN+halfN=N])
        # The original loop `range(halfN + 1)` went up to `i = halfN`.
        # For `i = halfN`, `data[:halfN + halfN]` which is `data[:N]`. This is a window of size N.
        # So `convolve` result for `mode='same'` should be fine if N is odd.
        # If N is even, `halfN = N/2`. `convolve` `same` mode pads symmetrically.
        # The custom edge handling below provides a specific behavior for these edges.
        # If N=4, halfN=2. i=0,1,2.
        # i=0: mean(data[:2])
        # i=1: mean(data[:3])
        # i=2: mean(data[:4])
        # Let's match the original intent as closely as possible:
        # Original loop went up to i = halfN (inclusive).
        # For data[i], window is data[0 ... i + halfN -1]
        start_idx_limit = halfN if N % 2 == 1 else halfN - 1  # Correct boundary for 'same' conv.
        for i in range(start_idx_limit):  # Boundary where 'same' convolution might not be using full desired window yet
            actual_win_end = min(len(data), i + halfN + (N % 2))  # Ensure it's within bounds
            outdata[i] = np.mean(data[max(0, i - halfN): actual_win_end])

    # Edge handling for the last halfN points
    # Original loop: for i in range(1, halfN + 1): outdata[-i] = np.mean(data[-i - halfN:])
    # Example N=4, halfN=2. i=1,2
    # i=1 (index -1): mean(data[-1-2 : ]) = mean(data[-3:])
    # i=2 (index -2): mean(data[-2-2 : ]) = mean(data[-4:])
    # This tries to make the window of size N, anchored at the end.
    for i in range(1, start_idx_limit + 1):
        actual_win_start = max(0, len(data) - i - halfN)
        outdata[-i] = np.mean(
            data[actual_win_start: len(data) - i + halfN + (N % 2) if N % 2 == 1 else len(data) - i + halfN])
        # Simpler interpretation from original:
        # For outdata[-i], the window is data[len(data) - i - halfN : len(data) - i + N - (len(data)-i-halfN)] ?
        # Original: data[-i-halfN:]
        # If i=1 (last element), data[-1-halfN:]. Window size can be > N here.
        # If i=halfN (Nth from end), data[-halfN-halfN:] = data[-N:]. Window size N.
        # This seems to want right-aligned windows of increasing size for the right edge.
        # Let's use a clearer definition for edge handling: use available data for window.
    for i in range(halfN):
        # For point `i` from the start
        win_end = min(len(data), i + halfN + 1)  # Window for point i includes up to N/2 points after it
        outdata[i] = np.mean(data[0:win_end])
        # For point `len(data) - 1 - i` from the end
        win_start = max(0, len(data) - 1 - i - halfN)
        outdata[len(data) - 1 - i] = np.mean(data[win_start:])

    return outdata


def print_progress(n: int | str) -> None:
    """
    Prints progress to the console, overwriting the previous line.

    Args:
        n: Number or string to print.
    """
    sys.stdout.write("\r" + str(n))
    sys.stdout.flush()


def phase_wrap(data: np.ndarray) -> np.ndarray:
    """
    Wraps phase angles to the interval [-pi, pi].

    Args:
        data: Array of phase angles in radians.

    Returns:
        Array with phase angles wrapped.
    """
    # np.angle(z) returns in [-pi, pi], so this is robust.
    return np.angle(np.exp(1j * data))


def fetch_timestamp_fast(timestamp_strs: list[str] | np.ndarray, downsampling: int = 100) -> tuple[
    list[datetime.datetime], np.ndarray]:
    """
    Efficiently converts a list of timestamp strings to datetime objects
    by parsing a downsampled subset and interpolating.

    Args:
        timestamp_strs: List or NumPy array of timestamp strings.
        downsampling: Factor by which to downsample timestamp strings for parsing.
                      If 0 or 1, all timestamps are parsed.

    Returns:
        ts: List of Python datetime objects.
        t: NumPy array of time in seconds from the first timestamp.
    """
    N = len(timestamp_strs)
    if N == 0:
        return [], np.array([])

    if downsampling <= 0:  # Avoid division by zero or negative
        downsampling = N + 1  # effectively parse all if N is small, or a large number to parse 1 if N is huge.
        # A better default would be to parse all if downsampling is invalid.
        num_sparse_points = N
    elif downsampling == 1:
        num_sparse_points = N
    else:
        num_sparse_points = max(2, N // downsampling)  # Ensure at least 2 points for interpolation if N > 0

    if N == 1:  # Single timestamp, no interpolation needed
        ts0 = parse(timestamp_strs[0])
        return [ts0], np.array([0.0])

    # Ensure num_sparse_points does not exceed N
    num_sparse_points = min(num_sparse_points, N)

    x_all = np.arange(N)

    # Select indices for sparse parsing
    if num_sparse_points == N:  # Parse all
        x_sparse_indices = x_all
    else:  # Downsample
        x_sparse_indices = np.round(np.linspace(0, N - 1, num_sparse_points)).astype(int)
        # Ensure unique indices, especially if N is small compared to num_sparse_points calculation
        x_sparse_indices = np.unique(x_sparse_indices)
        if len(x_sparse_indices) < 2 and N > 1:  # Need at least two unique points for interpolation
            # Fallback: use first and last if possible
            x_sparse_indices = np.array([0, N - 1], dtype=int) if N > 1 else np.array([0], dtype=int)
            x_sparse_indices = np.unique(x_sparse_indices)  # Recalc unique for N=1 case

    # Parse the sparse timestamps
    if isinstance(timestamp_strs, np.ndarray):
        ts_sparse_objects = [parse(ts_str) for ts_str in timestamp_strs[x_sparse_indices]]
    else:  # Is a list
        ts_sparse_objects = [parse(timestamp_strs[idx]) for idx in x_sparse_indices]

    ts_sparse_dt = np.array(ts_sparse_objects)

    if len(ts_sparse_dt) == 0:  # Should not happen if N > 0
        return [datetime.datetime.min] * N, np.full(N, np.nan)  # Placeholder error

    t0 = ts_sparse_dt[0]
    t_sparse_seconds = np.array([(t_obj - t0).total_seconds() for t_obj in ts_sparse_dt])

    # Interpolate time in seconds
    if len(x_sparse_indices) == 1:  # Only one sparse point (e.g., N=1 or extreme downsampling)
        t_all_seconds = np.full(N, t_sparse_seconds[0])
    else:
        t_all_seconds = np.interp(x_all, x_sparse_indices, t_sparse_seconds)

    # Convert all interpolated seconds back to datetime objects
    ts_all_datetime = [t0 + timedelta(seconds=sec) for sec in t_all_seconds]

    return ts_all_datetime, t_all_seconds


def multi_legend(lns: list, loc: str = 'best') -> None:
    """
    Creates a legend for multiple lines on a Matplotlib plot.

    Args:
        lns: List of Matplotlib line objects.
        loc: Location of the legend (e.g., 'best', 'upper right').
    """
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc=loc)


def datetime_interp(timex: list[datetime.datetime] | np.ndarray,
                    timex0: list[datetime.datetime] | np.ndarray,
                    y0: np.ndarray) -> np.ndarray:
    """
    Interpolates y-values (y0) at new time points (timex),
    based on original time points (timex0).

    Args:
        timex: New time points (datetime objects) at which to interpolate.
        timex0: Original time points (datetime objects) corresponding to y0.
        y0: Original y-values.

    Returns:
        Interpolated y-values at timex.
    """
    if not timex0 or not y0.size:
        return np.array([])  # Or np.full(len(timex), np.nan)

    # Convert datetime objects to seconds relative to the first original timestamp
    t_ref = timex0[0]
    x_numeric = np.array([(t - t_ref).total_seconds() for t in timex])
    x0_numeric = np.array([(t - t_ref).total_seconds() for t in timex0])

    if not x0_numeric.size:  # No reference points
        return np.full(len(timex), np.nan)
    if len(x0_numeric) == 1:  # Only one reference point, return its value for all new times
        return np.full(len(timex), y0[0])

    return np.interp(x_numeric, x0_numeric, y0)


def correlation_coefficient(a: np.ndarray, n: np.ndarray) -> float | None:
    """
    Calculates the Pearson correlation coefficient between two discrete time sequences.
    This is a NumPy-vectorized version of the original loop-based calculation.

    Args:
        a: First sequence.
        n: Second sequence.

    Returns:
        Pearson correlation coefficient, or None if lengths are mismatched or calculation fails.
    """
    if len(a) != len(n):
        # print("The length is wrong!") # Or raise ValueError
        return None
    if len(a) == 0:  # No data to correlate
        return None  # Or np.nan

    mean_a = np.mean(a)
    mean_n = np.mean(n)

    # Numerator: sum((a_i - mean_a) * (n_i - mean_n))
    term1 = np.sum((a - mean_a) * (n - mean_n))

    # Denominator part 1: sum((a_i - mean_a)^2)
    term2 = np.sum((a - mean_a) ** 2)
    # Denominator part 2: sum((n_i - mean_n)^2)
    term3 = np.sum((n - mean_n) ** 2)

    denominator_sqrt = np.sqrt(term2 * term3)

    if denominator_sqrt == 0:
        # This happens if one or both series have zero variance (e.g., are constant)
        # Pearson correlation is undefined or can be considered 0 or 1 depending on context.
        # np.corrcoef and scipy.stats.pearsonr would return NaN and may issue a warning.
        if term1 == 0:  # If numerator is also zero (e.g. both are perfectly flat and equal)
            return 1.0 if np.array_equal(a, n) else 0.0  # or np.nan
        return 0.0  # Or np.nan, to align with standard libraries
        # If one is flat and other varies, correlation is typically 0 or undefined.

    r = term1 / denominator_sqrt
    return r