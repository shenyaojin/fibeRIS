import pytest
import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from fiberis.analyzer.Data3D.core3D import Data3D

class TestData3D:
    @pytest.fixture
    def sample_data(self):
        # Create dummy 3D data (spatial x time)
        # 10 spatial points, 5 time steps
        data = np.random.rand(10, 5)
        taxis = np.linspace(0, 4, 5)
        xaxis = np.random.rand(10)
        yaxis = np.random.rand(10)
        
        d = Data3D(name="test_data_3d")
        d.data = data
        d.taxis = taxis
        d.xaxis = xaxis
        d.yaxis = yaxis
        d.variable_name = "pressure"
        return d

    def test_initialization(self):
        d = Data3D(name="init_test")
        assert d.name == "init_test"
        assert d.data is None
        assert d.history is not None

    def test_save_load_npz(self, sample_data, tmp_path):
        filename = tmp_path / "test_3d.npz"
        
        # Test save
        sample_data.savez(str(filename))
        assert filename.exists()
        
        # Test load
        new_d = Data3D()
        new_d.load_npz(str(filename))
        
        assert np.array_equal(new_d.data, sample_data.data)
        assert np.array_equal(new_d.taxis, sample_data.taxis)
        assert np.array_equal(new_d.xaxis, sample_data.xaxis)
        assert np.array_equal(new_d.yaxis, sample_data.yaxis)
        assert new_d.variable_name == sample_data.variable_name
        assert new_d.name == sample_data.name

    def test_save_validation(self):
        d = Data3D()
        # Should fail because data is None
        with pytest.raises(ValueError):
            d.savez("fail.npz")

    def test_info_str(self, sample_data):
        info = sample_data.get_info_str()
        assert "Data3D Object Summary" in info
        assert "pressure" in info
        assert "Data Shape: (10, 5)" in info
