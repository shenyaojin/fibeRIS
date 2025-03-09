import numpy as np
from fiberis.analyzer.io import reader_mariner_gauge1d
import numpy as np
import os, glob
# Shows the IO functions of fiberis - how do/should they work

# Load the path
GAUGEdata_path = 'data/legacy/s_well/gauge_data/' # Old files
output_path = 'data/new/s_well/gauge_data/' # Fiberis supported files
file_list = os.listdir(GAUGEdata_path)
file_path = [GAUGEdata_path + file for file in file_list]
for file in file_list:
    gauge_data = reader_mariner_gauge1d.MarinerGauge1D()
    # The read function should be able to read the old files
    gauge_data.read(GAUGEdata_path + file)
    # the write function can write the files in the fiberis format
    gauge_data.write(output_path + file)