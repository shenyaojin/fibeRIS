# Utils for reading HFTS2 data from h5 files
# Shenyao Jin, adopted from Peiyao Li, Ge Jin's codes

import numpy as np
from glob import glob
from fiberis.utils import io_utils
from dateutil import parser

from fiberis.io import core


# Need to self check function in coreT.py

class HFTS2DAS2D(core.DataIO):


    def read(self, folderpath):

        # Get folder info
        files = glob(folderpath + '/*.h5')
        files = np.sort(files)
        timestamps = np.array([parser.parse(f[-21:-4]) for f in files])

        # Check if start_time or end_time is a string and convert to datetime
        for file in files:
            raw_data, raw_data_time = io_utils.read_h5(file)
            # Write it to tmp npz file, then combine them


    def write(self, filename, *args): # TBD
        pass