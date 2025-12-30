import pytest
import numpy as np
import datetime
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from fiberis.utils import signal_utils

class TestSignalUtils:
    def test_samediff(self):
        data = np.array([1, 2, 4, 7, 11])
        diff = signal_utils.samediff(data)
        
        expected = np.array([1, 2, 3, 4, 4]) # Last element repeated
        assert np.array_equal(diff, expected)
        assert len(diff) == len(data)

    def test_samediff_empty(self):
        data = np.array([])
        diff = signal_utils.samediff(data)
        assert len(diff) == 0

    def test_samediff_single(self):
        data = np.array([5])
        diff = signal_utils.samediff(data)
        assert np.array_equal(diff, np.array([0.]))

    def test_fillnan(self):
        data = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        filled = signal_utils.fillnan(data)
        
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.allclose(filled, expected)
        assert not np.any(np.isnan(filled))

    def test_fillnan_no_nan(self):
        data = np.array([1.0, 2.0, 3.0])
        filled = signal_utils.fillnan(data)
        assert np.array_equal(filled, data)

    def test_fillnan_all_nan(self):
        data = np.array([np.nan, np.nan])
        filled = signal_utils.fillnan(data)
        assert np.all(np.isnan(filled))

    def test_timediff(self):
        ts1 = datetime.datetime(2023, 1, 1, 12, 0, 10)
        ts2 = datetime.datetime(2023, 1, 1, 12, 0, 0)
        
        diff = signal_utils.timediff(ts1, ts2)
        assert diff == 10.0

    def test_butter_filters_shapes(self):
        # Test that filter functions return coefficients of correct shape
        fs = 100.0
        b, a = signal_utils.butter_lppass(10.0, fs, order=2)
        assert len(b) == 3 # Order 2 -> 3 coeffs
        assert len(a) == 3

        b, a = signal_utils.butter_hppass(10.0, fs, order=2)
        assert len(b) == 3
        assert len(a) == 3

        b, a = signal_utils.butter_bandpass(5.0, 15.0, fs, order=2)
        assert len(b) == 5 # Bandpass doubles order effectively
        assert len(a) == 5

    def test_lpfilter_execution(self):
        # Create a signal with low and high freq components
        t = np.linspace(0, 1, 1000)
        dt = t[1] - t[0]
        low_freq = np.sin(2 * np.pi * 5 * t) # 5 Hz
        high_freq = np.sin(2 * np.pi * 50 * t) # 50 Hz
        signal = low_freq + high_freq
        
        # Filter out high freq (cutoff 10 Hz)
        filtered = signal_utils.lpfilter(signal, dt, freqcut=10.0)
        
        # Check correlation with low freq component
        # Should be high
        corr = np.corrcoef(filtered, low_freq)[0, 1]
        assert corr > 0.9

    def test_amp_spectrum(self):
        t = np.linspace(0, 1, 1000, endpoint=False)
        dt = t[1] - t[0]
        freq = 10.0
        signal = np.sin(2 * np.pi * freq * t)
        
        freqs, amps = signal_utils.amp_spectrum(signal, dt)
        
        # Find peak frequency
        peak_idx = np.argmax(amps)
        peak_freq = freqs[peak_idx]
        
        assert np.isclose(peak_freq, freq, atol=1.0)
