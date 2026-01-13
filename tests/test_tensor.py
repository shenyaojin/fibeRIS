import pytest
import numpy as np
import datetime
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from fiberis.analyzer.TensorProcessor.coreT import CoreTensor

class TestCoreTensor:
    @pytest.fixture
    def sample_tensor_2d(self):
        # Create a 2D tensor series (2x2 matrices over 5 time steps)
        # Identity matrix at each step
        data = np.zeros((2, 2, 5))
        for i in range(5):
            data[:, :, i] = np.eye(2)
            
        taxis = np.linspace(0, 4, 5)
        start_time = datetime.datetime(2023, 1, 1, 12, 0, 0)
        
        return CoreTensor(data=data, taxis=taxis, dim=2, start_time=start_time, name="test_tensor_2d")

    def test_initialization(self, sample_tensor_2d):
        assert sample_tensor_2d.data.shape == (2, 2, 5)
        assert sample_tensor_2d.dim == 2
        assert len(sample_tensor_2d.taxis) == 5
        assert sample_tensor_2d.name == "test_tensor_2d"

    def test_set_data_validation(self, sample_tensor_2d):
        # Valid update
        new_data = np.zeros((2, 2, 5))
        sample_tensor_2d.set_data(new_data)
        assert np.array_equal(sample_tensor_2d.data, new_data)

        # Invalid dimension (3x3 instead of 2x2)
        bad_dim_data = np.zeros((3, 3, 5))
        with pytest.raises(ValueError):
            sample_tensor_2d.set_data(bad_dim_data)

        # Invalid time length
        bad_time_data = np.zeros((2, 2, 6))
        with pytest.raises(ValueError):
            sample_tensor_2d.set_data(bad_time_data)

    def test_rotate_tensor_2d(self, sample_tensor_2d):
        # Rotate identity matrix by 90 degrees (pi/2)
        # R = [[0, -1], [1, 0]]
        # R * I * R^T = R * R^T = I (Rotation of identity is identity? No wait)
        # R * I * R^T = [[0, -1], [1, 0]] * [[0, 1], [-1, 0]] = [[1, 0], [0, 1]] = I
        # Identity matrix is invariant under rotation.
        
        sample_tensor_2d.rotate_tensor(np.pi / 2)
        expected = np.eye(2)
        for i in range(5):
            assert np.allclose(sample_tensor_2d.data[:, :, i], expected)

        # Let's try a non-identity matrix: [[1, 0], [0, 0]] (stress in x only)
        # Rotate 90 deg -> stress in y only: [[0, 0], [0, 1]]
        data = np.zeros((2, 2, 1))
        data[0, 0, 0] = 1.0
        taxis = np.array([0.0])
        ct = CoreTensor(data=data, taxis=taxis, dim=2)
        
        ct.rotate_tensor(np.pi / 2)
        
        expected = np.array([[0.0, 0.0], [0.0, 1.0]])
        assert np.allclose(ct.data[:, :, 0], expected, atol=1e-10)

    def test_save_load_npz(self, sample_tensor_2d, tmp_path):
        filename = tmp_path / "test_tensor.npz"
        sample_tensor_2d.savez(str(filename))
        
        new_ct = CoreTensor()
        new_ct.load_npz(str(filename))
        
        assert np.array_equal(new_ct.data, sample_tensor_2d.data)
        assert np.array_equal(new_ct.taxis, sample_tensor_2d.taxis)
        assert new_ct.dim == sample_tensor_2d.dim
        assert new_ct.start_time == sample_tensor_2d.start_time

    def test_unsupported_rotation(self):
        # 4D tensor not supported
        data = np.zeros((4, 4, 1))
        taxis = np.array([0.0])
        ct = CoreTensor(data=data, taxis=taxis, dim=4)
        
        with pytest.raises(ValueError):
            ct.rotate_tensor(0.1)
