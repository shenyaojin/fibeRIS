# Utils for reading Mariner 1D pumping data
# Shenyao Jin
# I have already packed the data into .npz files, so the .read function is simply reading the
# data from those .npz files.

import numpy as np
from fiberis.io import core
from fiberis.analyzer.Data1D import Data1DPumpingCurve
import os

class MarinerPP1D(core.DataIO):

    def __init__(self):
        """
        Initialize the PumPing data reader
        """
        super().__init__()
        self.label = None

    def read(self, filename: str, label: str):
        """
        Read a specific data series from the pumping data npz file.

        The npz file is expected to contain a 'value' array (2D) and a 'label' array (1D).
        This method will find the specified label and load the corresponding data row.

        :param filename: The filename of the npz file.
        :param label: The label of the data series to read (e.g., 'SLURRY_RATE').
        """
        self.filename = filename
        data_structure = np.load(filename, allow_pickle=True)

        all_labels = list(data_structure['label'])
        all_values = data_structure['value']

        try:
            target_index = all_labels.index(label)
        except ValueError:
            raise ValueError(f"Label '{label}' not found in file. Available labels: {all_labels}")

        self.data = all_values[target_index]
        self.label = label

        taxis_tmp = data_structure['taxis']
        # calculate the time axis for taxis_tmp is in datetime.datetime format
        self.taxis = np.zeros_like(taxis_tmp, dtype=float)
        self.start_time = taxis_tmp[0]
        for i in range(len(taxis_tmp)):
            self.taxis[i] = (taxis_tmp[i] - self.start_time).total_seconds()

    def write(self, filename: str, **kwargs):
        """
        Write the loaded data series to a standard .npz file.

        :param filename: The filename of the npz file to save.
        :param kwargs: Reserved for future format options.
        """
        if self.data is None or self.label is None:
            raise ValueError("Data has not been loaded. Call read() with a specific label first.")

        if not filename.endswith('.npz'):
            filename += '.npz'

        np.savez(
            filename,
            data=self.data,
            taxis=self.taxis,
            start_time=self.start_time,
            label=self.label
        )

    def to_analyzer(self) -> Data1DPumpingCurve:
        """
        Directly creates and populates a Data1DPumpingCurve analyzer object.

        Returns:
            Data1DPumpingCurve: A populated analyzer object ready for use.
        """
        if self.data is None or self.taxis is None or self.start_time is None:
            raise ValueError("Data is not loaded. Please call read() first.")

        analyzer = Data1DPumpingCurve()
        analyzer.data = self.data
        analyzer.taxis = self.taxis
        analyzer.start_time = self.start_time
        analyzer.name = self.label

        analyzer.history.add_record(f"Data populated from {self.__class__.__name__}.")

        return analyzer