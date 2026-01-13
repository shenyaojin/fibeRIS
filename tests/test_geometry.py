import pytest
import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from fiberis.analyzer.Data1DG.core1DG import Data1DG
from fiberis.analyzer.Geometry3D.coreG3D import DataG3D

class TestData1DG:
    @pytest.fixture
    def sample_data(self):
        daxis = np.linspace(0, 10, 11)
        data = np.sin(daxis)
        return Data1DG(data=data, daxis=daxis, axis_name="Depth (m)", name="test_1dg")

    def test_initialization(self, sample_data):
        assert sample_data.data is not None
        assert sample_data.daxis is not None
        assert sample_data.axis_name == "Depth (m)"
        assert len(sample_data.data) == 11

    def test_select_range(self, sample_data):
        sample_data.select_range(2.0, 5.0)
        assert len(sample_data.data) == 4 # 2, 3, 4, 5
        assert sample_data.daxis[0] == 2.0
        assert sample_data.daxis[-1] == 5.0

    def test_shift(self, sample_data):
        sample_data.shift(5.0)
        assert sample_data.daxis[0] == 5.0
        assert sample_data.daxis[-1] == 15.0

    def test_get_value_by_location(self, sample_data):
        # Test exact location
        val = sample_data.get_value_by_location(np.pi/2) # ~1.57
        # sin(pi/2) = 1. Linear interp between 1 and 2 might not be exactly 1 but close
        # daxis has integer steps. 1->sin(1)~0.84, 2->sin(2)~0.91. 
        # 1.57 is between 1 and 2.
        
        # Let's test a known point
        val_at_2 = sample_data.get_value_by_location(2.0)
        assert np.isclose(val_at_2, np.sin(2.0))

    def test_save_load_npz(self, sample_data, tmp_path):
        filename = tmp_path / "test_1dg.npz"
        sample_data.savez(str(filename))
        
        new_d = Data1DG()
        new_d.load_npz(str(filename))
        
        assert np.array_equal(new_d.data, sample_data.data)
        assert np.array_equal(new_d.daxis, sample_data.daxis)
        assert new_d.axis_name == sample_data.axis_name


class TestDataG3D:
    @pytest.fixture
    def sample_data(self):
        # Create a simple 3D line: x=t, y=0, z=0
        t = np.linspace(0, 10, 11)
        d = DataG3D()
        d.xaxis = t
        d.yaxis = np.zeros_like(t)
        d.zaxis = np.zeros_like(t)
        d.data = np.zeros_like(t) # Placeholder
        d.name = "test_g3d"
        return d

    def test_calculate_md(self, sample_data):
        md = sample_data.calculate_md()
        # Since it's a straight line along x, MD should equal x
        assert np.allclose(md, sample_data.xaxis)
        
        # Test diagonal: x=t, y=t, z=0. Dist = sqrt(2)*t
        sample_data.yaxis = sample_data.xaxis
        md_diag = sample_data.calculate_md()
        expected = sample_data.xaxis * np.sqrt(2)
        assert np.allclose(md_diag, expected)

    def test_save_load_npz(self, sample_data, tmp_path):
        filename = tmp_path / "test_g3d.npz"
        sample_data.savez(str(filename))
        
        new_d = DataG3D()
        new_d.load_npz(str(filename))
        
        assert np.array_equal(new_d.xaxis, sample_data.xaxis)
        assert np.array_equal(new_d.yaxis, sample_data.yaxis)
        assert np.array_equal(new_d.zaxis, sample_data.zaxis)
        assert new_d.name == "test_g3d"

    def test_save_validation(self):
        d = DataG3D()
        with pytest.raises(ValueError):
            d.savez("fail.npz")
