import pytest
import numpy as np
import time
import datetime
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from fiberis.analyzer.Data1D.core1D import Data1D
from fiberis.analyzer.Data2D.core2D import Data2D
from fiberis.analyzer.TensorProcessor.coreT import CoreTensor

class TestPerformance:
    def test_data1d_large_crop(self):
        # Create 1 million points
        N = 1_000_000
        data = np.random.rand(N)
        taxis = np.linspace(0, 1000, N)
        start_time = datetime.datetime(2023, 1, 1, 0, 0, 0)
        d = Data1D(data=data, taxis=taxis, start_time=start_time, name="perf_1d")
        
        # Benchmark crop
        start_time = time.time()
        d.crop(100.0, 900.0)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"\nData1D crop (1M points): {duration:.4f} seconds")
        
        # Assert it's reasonably fast (e.g., < 0.5s)
        assert duration < 0.5

    def test_data2d_large_init(self):
        # Create 1000x1000 grid
        rows, cols = 1000, 1000
        data = np.random.rand(rows, cols)
        taxis = np.linspace(0, 100, cols)
        daxis = np.linspace(0, 100, rows)
        
        start_time = time.time()
        d = Data2D(data=data, taxis=taxis, daxis=daxis, name="perf_2d")
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"\nData2D init (1000x1000): {duration:.4f} seconds")
        assert duration < 0.5

    def test_tensor_rotation_performance(self):
        # 10,000 time steps of 3x3 tensors
        steps = 10_000
        data = np.random.rand(3, 3, steps)
        taxis = np.linspace(0, 100, steps)
        ct = CoreTensor(data=data, taxis=taxis, dim=3, name="perf_tensor")
        
        start_time = time.time()
        ct.rotate_tensor(np.pi/4)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"\nTensor rotation (10k steps): {duration:.4f} seconds")
        
        # Matrix multiplication loop in python might be slow, but numpy is optimized
        # 10k 3x3 matmuls should be fast
        assert duration < 1.0
