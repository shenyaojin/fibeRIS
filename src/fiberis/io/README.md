The scripts in `fiberis.io` are designed to load file from various sources, then 
store them in a uniform format in a `.npz` file.

This I/O package would be redesigned in the future for better performance. 

The logical chain would be:

Original file (different formats) -> Specific type of data (e.g. pressure gauge data, fiber optical data, etc.) -> Uniform format (e.g. `.npz` file, the 
only difference would be the dimensions of the data) -> Data analysis

The `fiberis.io` package is designed to convert the original file to a specific type of data, and then store it in a uniform format.

So in the package every file is a specific type of data format, and use the `self.load_data()` method to load the data from the file then store it in 
`fiberis.analyzer.xxx` format. 

Then another function `self.save_data()` should be used (and it should be in `fiberis.analyzer.xxx`) to save the data in a uniform format `.npz` file.

