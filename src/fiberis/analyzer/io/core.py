# This file contains the core class for data input/output.
# Read data from different types of file(s)
# Write data npz(or csv) to make it faster to read

import importlib
import pkgutil
import os

# dynamically import all modules in the io package
# package = 'fiberis.analyzer.io'
# package_dir = os.path.dirname(__file__)
# # filter the modules starting with "reader_"
# modules = [name for _, name, _ in pkgutil.iter_modules([package_dir])]
#
# # import all modules in the io package
# for module in modules:
#     module_name = f'{package}.{module}'
#     importlib.import_module(module_name)
#     print(f'Imported {module_name}')

class dataio():

    # constructor
    def __init__(self):
        """
        Initialize the dataio object.
        """
        pass

    @abstractmethod
    def read(self, filename):
        """
        Read data from a file.

        Parameters:
        ----------
        filename : str
            The path to the file containing the data.
        """
        pass

    @abstractmethod
    def write(self, filename, *args):
        """
        Write data to a file.

        Parameters:
        ----------
        filename : str
            The path to the file to write the data.
        data : numpy.ndarray
            The data to write.
        """
        pass