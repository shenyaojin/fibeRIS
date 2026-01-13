import pytest
import numpy as np
import datetime
import os
import sys

# Add src to path so we can import fiberis
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from fiberis.analyzer.Data1D.core1D import Data1D

class TestData1D:
    @pytest.fixture
    def sample_data(self):
        taxis = np.linspace(0, 10, 11)  # 0, 1, ..., 10
        data = np.sin(taxis)
        start_time = datetime.datetime(2023, 1, 1, 12, 0, 0)
        return Data1D(data=data, taxis=taxis, start_time=start_time, name="test_data")

    def test_initialization(self, sample_data):
        assert sample_data.data is not None
        assert sample_data.taxis is not None
        assert sample_data.start_time == datetime.datetime(2023, 1, 1, 12, 0, 0)
        assert sample_data.name == "test_data"
        assert len(sample_data.data) == 11

    def test_crop_datetime(self, sample_data):
        start_crop = datetime.datetime(2023, 1, 1, 12, 0, 2)
        end_crop = datetime.datetime(2023, 1, 1, 12, 0, 5)
        
        sample_data.crop(start_crop, end_crop)
        
        assert len(sample_data.data) == 4  # 2, 3, 4, 5
        assert sample_data.taxis[0] == 0.0
        assert sample_data.start_time == start_crop
        assert np.isclose(sample_data.data[0], np.sin(2))

    def test_crop_float(self, sample_data):
        start_crop = 2.0
        end_crop = 5.0
        
        sample_data.crop(start_crop, end_crop)
        
        assert len(sample_data.data) == 4
        assert sample_data.taxis[0] == 0.0
        # Start time should shift by 2 seconds
        expected_start = datetime.datetime(2023, 1, 1, 12, 0, 2)
        assert sample_data.start_time == expected_start

    def test_crop_invalid_range(self, sample_data):
        with pytest.raises(ValueError):
            sample_data.crop(5.0, 2.0)

    def test_crop_empty_result(self, sample_data):
        sample_data.crop(20.0, 30.0)
        assert len(sample_data.data) == 0
        assert len(sample_data.taxis) == 0

    def test_right_merge(self, sample_data):
        # Create a second dataset starting after the first one
        taxis2 = np.linspace(0, 5, 6)
        data2 = np.cos(taxis2)
        # First data ends at 10s (12:00:10). Let's start next one at 12:00:15
        start_time2 = datetime.datetime(2023, 1, 1, 12, 0, 15)
        other_data = Data1D(data=data2, taxis=taxis2, start_time=start_time2, name="data2")
        
        original_len = len(sample_data.data)
        sample_data.right_merge(other_data)
        
        assert len(sample_data.data) == original_len + len(data2)
        # Check if time gap is preserved
        # Last point of original was at 10s. New data starts at 15s relative to original start.
        # So first point of appended data should be at 15.0
        assert np.isclose(sample_data.taxis[original_len], 15.0)

    def test_remove_abnormal_data_mean(self, sample_data):
        # Introduce a spike
        sample_data.data[5] = 1000.0
        sample_data.remove_abnormal_data(threshold=10.0, method='mean')
        
        expected_val = (sample_data.data[4] + sample_data.data[6]) / 2.0
        assert np.isclose(sample_data.data[5], expected_val)

    def test_remove_abnormal_data_nan(self, sample_data):
        sample_data.data[5] = 1000.0
        sample_data.remove_abnormal_data(threshold=10.0, method='nan')
        assert np.isnan(sample_data.data[5])

    def test_load_npz(self, tmp_path):
        # Create a dummy npz file
        filename = tmp_path / "test.npz"
        data = np.array([1.0, 2.0, 3.0])
        taxis = np.array([0.0, 1.0, 2.0])
        start_time = datetime.datetime(2023, 1, 1, 10, 0, 0)
        
        np.savez(filename, data=data, taxis=taxis, start_time=start_time)
        
        d = Data1D()
        d.load_npz(str(filename))
        
        assert np.array_equal(d.data, data)
        assert np.array_equal(d.taxis, taxis)
        assert d.start_time == start_time
        assert d.name == "test.npz"

    def test_load_npz_missing_key(self, tmp_path):
        filename = tmp_path / "bad.npz"
        np.savez(filename, data=np.array([1])) # Missing taxis and start_time
        
        d = Data1D()
        with pytest.raises(KeyError):
            d.load_npz(str(filename))
