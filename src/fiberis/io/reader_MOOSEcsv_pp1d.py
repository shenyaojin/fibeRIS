# MOOSE generated CSV reader for 1D data.
# Shenyao Jin, shenyaojin@mines.edu, 05/18/2025
import numpy as np
import pandas as pd
import datetime
import os
from fiberis.io import core


class MOOSEcsv_pp1d(core.DataIO):
    """
    DataIO class for reading specific columns from MOOSE CSV files
    and writing them to NPZ format.
    """

    def __init__(self):
        """
        Initializes the MOOSEcsv_pp1d object.
        Sets up placeholders for time axis, data, labels, and a default start time.
        """
        super().__init__()  # Call the parent class's constructor
        self.taxis = None  # Will store the time data (numpy array)
        self.data = []  # Will store a list of data arrays (e.g., [np.array_for_key1])
        self.label = []  # Will store a list of labels for the data arrays (e.g., ["key1"])
        self.start_time = datetime.datetime(2024, 1, 1)  # Default start time

    def list_available_keys(self, filename=None):
        """
        Reads and prints the available column headers (keys) from the specified CSV file.
        This helps in identifying which keys can be used with the read() method.
        The 'time' column is identified separately.

        :param filename: str, path to the CSV file.
        :raises ValueError: If filename is not provided.
        :raises FileNotFoundError: If the specified CSV file does not exist.
        :raises Exception: For other pandas CSV reading errors.
        :return: list, a list of available data keys (column headers excluding 'time').
                 Returns an empty list if an error occurs or no data keys are found.
        """
        if filename is None:
            raise ValueError("Filename must be provided to list available keys.")

        try:
            # Read only the header row to get column names efficiently
            headers = pd.read_csv(filename, nrows=0).columns.tolist()
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
            raise
        except pd.errors.EmptyDataError:
            print(f"Error: File '{filename}' is empty or contains no data.")
            return []
        except Exception as e:
            print(f"Error reading CSV file '{filename}' to get headers: {e}")
            raise

        if not headers:
            print(f"No headers found in the CSV file '{filename}'.")
            return []

        print(f"\nAvailable columns in '{filename}':")
        data_keys = []
        if 'time' in headers:
            print(f"  - time (Time axis column)")
        else:
            print("  Warning: 'time' column not found. Expected for this reader.")

        for header in headers:
            if header != 'time':
                print(f"  - {header} (Data key)")
                data_keys.append(header)

        if not data_keys and 'time' in headers and len(headers) == 1:
            print("  No additional data keys found besides the 'time' column.")
        elif not data_keys:
            print("  No data keys found.")

        print("-" * 30)  # Separator
        return data_keys

    def read(self, filename=None, key=None):
        """
        Reads the 'time' column and a specified data column ('key') from a MOOSE CSV file.

        The data is stored in the instance attributes:
        - self.taxis: NumPy array of time values.
        - self.data: A list containing a single NumPy array for the specified key's data.
        - self.label: A list containing the key string.

        :param filename: str, path to the CSV file.
        :param key: str, the header name of the data column to read (e.g., "pp_mon2").
        :raises ValueError: If filename or key is not provided, or if 'time' column
                            or the specified key is not found in the CSV.
        :raises FileNotFoundError: If the specified CSV file does not exist.
        :raises Exception: For other pandas CSV reading errors.
        :return: None
        """
        if filename is None:
            raise ValueError("Filename must be provided.")
        if key is None:
            raise ValueError("Data 'key' must be provided to specify which column to read.")

        try:
            df = pd.read_csv(filename)
        except FileNotFoundError:
            # Let the original error propagate for more details
            # print(f"Error: File '{filename}' not found.")
            raise
        except Exception as e:
            # print(f"Error reading CSV file '{filename}': {e}")
            raise  # Or handle more gracefully

        if 'time' not in df.columns:
            raise ValueError("CSV file must contain a 'time' column.")

        if key not in df.columns:
            raise ValueError(f"Key '{key}' not found in CSV columns. Available columns: {df.columns.tolist()}")

        self.taxis = df['time'].to_numpy()
        # Store data for the specified key as a list containing one NumPy array
        self.data = [df[key].to_numpy()]
        # Store the key as a list containing one string label
        self.label = [key]

        return  # As specified in the initial template

    def write(self, filename, **kwargs):
        """
        Write the loaded data to an NPZ file. If multiple keys were loaded (not current
        behavior of read), it would write multiple NPZ files. With the current 'read'
        method (single key), it writes one NPZ file.

        Each NPZ file is named using the provided base filename and the corresponding label.
        Example: if filename="output/data" and label="pp_mon2", file will be "output/data_pp_mon2.npz".

        :param filename: The base filename (path prefix) to save the NPZ file(s).
                         If it ends with '.npz', the extension is removed first.
        :param kwargs: Reserved for future format options. Currently unused.
        :raises ValueError: If data, labels, or taxis are not properly initialized
                            (e.g., if read() has not been called successfully).
        :return: None
        """
        # Ensure filename does not end with '.npz' as we'll add it with the label
        if filename.endswith('.npz'):
            filename = filename[:-4]

        # Validate that data has been loaded
        if not self.data or not self.label:  # Checks if lists are empty
            raise ValueError("Data and labels are not initialized. Call read() first.")
        if len(self.label) != len(self.data):
            # This check is important if read() could load multiple keys
            raise ValueError("Mismatch between the number of labels and data arrays.")
        if self.taxis is None:
            raise ValueError("Time axis (taxis) is not initialized. Call read() first.")

        # Ensure output directory exists if filename includes a path
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)  # Create directory if it doesn't exist

        # Save each label's data to a separate npz file
        # (Currently, there will be only one label and one data array)
        for i, lbl in enumerate(self.label):
            # Construct the filename for this label (e.g., basefilename_label.npz)
            label_filename = f"{filename}_{lbl}.npz"

            # Extract the data array for the current label
            data_for_label = self.data[i]

            # Save the data along with time axis and start time
            np.savez(
                label_filename,
                data=data_for_label,
                taxis=self.taxis,
                start_time=self.start_time,  # datetime.datetime object
                label=lbl  # Store the label itself in the npz for context
            )
            # You can add a print statement for confirmation if desired:
            # print(f"Data for '{lbl}' saved to '{label_filename}'")

        return