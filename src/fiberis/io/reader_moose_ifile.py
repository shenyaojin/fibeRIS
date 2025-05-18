# This file is part of MOOSE I/O module.
# Purpose: Read MOOSE .i input files and store them in fibeRIS data format.
# Shenyao Jin, 05/13/2025

from fiberis.io import core


class MOOSEIFileReader(core.DataIO):

    def __init__(self, fname):
        """
        :param fname: Name of the MOOSE input file. Can be the single file name, or the .i file (full name).
        Initialize the MOOSE I/O
        """
        self.fname = fname

    def read(self, **kwargs):

        return

    def write(self, filename, *args):

        return
