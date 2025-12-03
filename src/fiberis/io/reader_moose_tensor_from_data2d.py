# fibeRIS/src/fiberis/io/reader_moose_tensor_from_data2d.py
import numpy as np
from fiberis.io.core import DataIO
from fiberis.analyzer.Data2D.core2D import Data2D
from fiberis.analyzer.TensorProcessor.coreT import CoreTensor
from typing import List

class MOOSETensorFromData2D(DataIO):
    def __init__(self):
        super().__init__()
        self.tensor_data_list: List[CoreTensor] = []

    def read(self, strain_xx: Data2D, strain_yy: Data2D, strain_xy: Data2D):
        
        if not (strain_xx.data.shape == strain_yy.data.shape == strain_xy.data.shape):
            raise ValueError("Input Data2D objects must have the same shape.")

        num_points = strain_xx.data.shape[0]
        num_times = strain_xx.data.shape[1]

        self.taxis = strain_xx.taxis
        self.daxis = strain_xx.daxis
        self.start_time = strain_xx.start_time

        for i in range(num_points):
            tensor_over_time = np.zeros((2, 2, num_times))
            tensor_over_time[0, 0, :] = strain_xx.data[i, :]
            tensor_over_time[1, 1, :] = strain_yy.data[i, :]
            tensor_over_time[0, 1, :] = strain_xy.data[i, :]
            tensor_over_time[1, 0, :] = strain_xy.data[i, :] # Assuming symmetric tensor

            tensor_obj = CoreTensor(
                data=tensor_over_time,
                taxis=self.taxis,
                dim=2,
                start_time=self.start_time,
                name=f"Tensor at daxis={self.daxis[i]}"
            )
            self.tensor_data_list.append(tensor_obj)

    def to_analyzer(self):
        return self.tensor_data_list
        
    def write(self, filename, *args):
        pass # Not implemented for now
