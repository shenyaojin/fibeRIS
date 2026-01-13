import pytest
import numpy as np
import datetime
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from fiberis.analyzer.Data2D.core2D import Data2D

class TestData2D:
    @pytest.fixture
    def sample_data(self):
        # Create a 10x5 data array (10 depth points, 5 time points)
        data = np.random.rand(10, 5)
        taxis = np.linspace(0, 4, 5)
        daxis = np.linspace(0, 9, 10)
        start_time = datetime.datetime(2023, 1, 1, 12, 0, 0)
        return Data2D(data=data, taxis=taxis, daxis=daxis, start_time=start_time, name="test_data_2d")

    def test_initialization(self, sample_data):
        assert sample_data.data is not None
        assert sample_data.taxis is not None
        assert sample_data.daxis is not None
        assert sample_data.start_time == datetime.datetime(2023, 1, 1, 12, 0, 0)
        assert sample_data.name == "test_data_2d"
        assert sample_data.data.shape == (10, 5)

    def test_set_data_validation(self, sample_data):
        # Test valid update
        new_data = np.random.rand(10, 5)
        sample_data.set_data(new_data)
        assert np.array_equal(sample_data.data, new_data)

        # Test invalid shape (time dimension mismatch)
        bad_data_time = np.random.rand(10, 6)
        with pytest.raises(ValueError):
            sample_data.set_data(bad_data_time)

        # Test invalid shape (depth dimension mismatch)
        bad_data_depth = np.random.rand(11, 5)
        with pytest.raises(ValueError):
            sample_data.set_data(bad_data_depth)

    def test_set_taxis_validation(self, sample_data):
        # Test valid update
        new_taxis = np.linspace(0, 4, 5)
        sample_data.set_taxis(new_taxis)
        assert np.array_equal(sample_data.taxis, new_taxis)

        # Test invalid length
        bad_taxis = np.linspace(0, 5, 6)
        with pytest.raises(ValueError):
            sample_data.set_taxis(bad_taxis)

    def test_set_daxis_validation(self, sample_data):
        # Test valid update
        new_daxis = np.linspace(0, 9, 10)
        sample_data.set_daxis(new_daxis)
        assert np.array_equal(sample_data.daxis, new_daxis)

        # Test invalid length
        bad_daxis = np.linspace(0, 10, 11)
        with pytest.raises(ValueError):
            sample_data.set_daxis(bad_daxis)

    def test_remove_timezone(self):
        tz_aware = datetime.datetime(2023, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
        d = Data2D(start_time=tz_aware)
        assert d.start_time.tzinfo is not None
        
        d.remove_timezone()
        assert d.start_time.tzinfo is None
        assert d.start_time.year == 2023

    def test_load_save_npz(self, sample_data, tmp_path):
        filename = tmp_path / "test_2d.npz"
        
        # Save manually first to test load
        np.savez(filename, 
                 data=sample_data.data, 
                 taxis=sample_data.taxis, 
                 daxis=sample_data.daxis, 
                 start_time=sample_data.start_time)
        
        new_d = Data2D()
        new_d.load_npz(str(filename))
        
        assert np.array_equal(new_d.data, sample_data.data)
        assert np.array_equal(new_d.taxis, sample_data.taxis)
        assert np.array_equal(new_d.daxis, sample_data.daxis)
        assert new_d.start_time == sample_data.start_time

    def test_rename(self, sample_data):
        sample_data.rename("new_name")
        assert sample_data.name == "new_name"
        
        sample_data.set_filename("part1", "part2")
        assert sample_data.name == "part1_part2"
