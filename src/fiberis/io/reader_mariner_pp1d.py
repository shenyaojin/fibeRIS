# Utils for reading Mariner 1D pumping data
# Shenyao Jin
# I have already packed the data into .npz files, so the .read function is simply reading the
# data from those .npz files.

import numpy as np
from fiberis.io import core

class MarinerPP1D(core.DataIO):

    def __init__(self):
        """
        Initialize the PumPing data reader
        """
        self.taxis = None
        self.data = None
        self.start_time = None
        self.label = None

    def read(self, filename=None):
        """
        Read the pumping data from the npz file. If you want to set the data manually,
        you can use the set_data method.
        :param filename: the filename of the npz file
        :return: None
        """

        data_structure = np.load(filename, allow_pickle=True)
        self.data = data_structure['value']

        taxis_tmp = data_structure['taxis']
        # calculate the time axis for taxis_tmp is in datetime.datetime format
        self.taxis = np.zeros_like(taxis_tmp, dtype=float)
        self.start_time = taxis_tmp[0]
        for i in range(len(taxis_tmp)):
            self.taxis[i] = (taxis_tmp[i] - self.start_time).total_seconds()

        self.label = data_structure['label']

    def write(self, filename, **kwargs):
        """
        Write the pumping data to multiple npz files, each named using the provided filename and the corresponding label.

        :param filename: the base filename (without extension) to save the npz files
        :param kwargs: format options (reserved for future expansion)
        :return: None
        """

        # Ensure filename does not end with '.npz' since multiple files will be created
        if filename.endswith('.npz'):
            filename = filename[:-4]

        # Validate data and labels
        if self.data is None or self.label is None or len(self.label) != self.data.shape[0]:
            raise ValueError("Data and labels must be properly initialized and match in length.")

        # Save each label's data to a separate npz file
        for i, lbl in enumerate(self.label):
            # Construct the filename for this label
            label_filename = f"{filename}{lbl}.npz"

            # Extract the data for the current label
            data_for_label = self.data[i]

            # Save the data along with other necessary information
            np.savez(
                label_filename,
                data=data_for_label,
                taxis=self.taxis,
                start_time=self.start_time,
            )

