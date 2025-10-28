# History log system for Data Processing and Analysis
# Shenyao Jin, shenyaojin@mines.edu
# 03/09/2025
# Docs finished by Gemini, 06/03/2025

import datetime
import os
from typing import List, Dict, Optional, Callable, Any


class InfoManagementSystem:
    """
    A simplified information management class that stores records with
    automatic timestamps and optional severity levels.

    Attributes:
    -----------
    records : list of dict
        A list of record dictionaries, each containing a 'description',
        'timestamp', and 'level'.
    """

    def __init__(self):
        """
        Initializes the system with an empty list of records.
        """
        self.records: List[Dict[str, Any]] = []

    def add_record(self, description: str, level: str = "INFO") -> None:
        """
        Creates a new record with a given description, severity level,
        and the current timestamp, then appends it to records.

        Parameters
        ----------
        description : str
            The description or information to be stored in the record.
        level : str, optional
            The severity level of the record (e.g., "INFO", "WARNING", "ERROR").
            Defaults to "INFO".
        """
        if not isinstance(description, str):
            # Basic type check to prevent non-string descriptions
            description = str(description)

        if not isinstance(level, str):
            level = str(level).upper()
        else:
            level = level.upper()  # Standardize level to uppercase

        new_record = {
            'description': description,
            'timestamp': datetime.datetime.now(),
            'level': level
        }
        self.records.append(new_record)

    def get_records(self, filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None) -> List[Dict[str, Any]]:
        """
        Retrieves stored records, optionally filtered by a user-defined
        function.

        Parameters
        ----------
        filter_fn : callable, optional
            A function that takes a single record (dict) and returns a
            boolean. If provided, only records for which filter_fn(record)
            is True will be returned. Defaults to None (no filtering).

        Returns
        -------
        list of dict
            All or filtered records.
        """
        if filter_fn is None:
            return list(self.records)  # Return a copy
        else:
            return [rec for rec in self.records if filter_fn(rec)]

    def print_records(self, filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None) -> None:
        """
        Prints each record, optionally filtered by a user-defined function.
        Records are printed with their level, timestamp, and description.

        Parameters
        ----------
        filter_fn : callable, optional
            A function that takes a single record (dict) and returns a
            boolean. If provided, only records for which filter_fn(record)
            is True will be printed. Defaults to None (no filtering).
        """
        filtered_records = self.get_records(filter_fn)
        if not filtered_records:
            print("No records to display.")
            return

        for rec in filtered_records:
            timestamp_str = rec['timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Millisecond precision
            print(f"[{rec.get('level', 'N/A')}] {timestamp_str}: {rec['description']}")

    def save_records_to_txt(self, save_path: Optional[str] = None,
                            filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None) -> str:
        """
        Saves the records to a .txt file, optionally filtered.

        If no save_path is provided, it saves the file in the current working
        directory with a timestamped filename. If a directory is provided,
        it saves the file inside that directory with the same naming convention.

        Parameters
        ----------
        save_path : str, optional
            The directory or full file path where the log should be saved.
            If a directory is given, a timestamped filename will be generated.
            If None, the current working directory is used.
        filter_fn : callable, optional
            A function to filter which records are saved.

        Returns
        -------
        str
            The path to the saved file, or an empty string if saving failed.
        """
        filtered_records = self.get_records(filter_fn)
        if not filtered_records:
            print("No records to save.")
            return ""

        if save_path and os.path.isdir(save_path):
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = f"history_log_{timestamp}.txt"
            filepath = os.path.join(save_path, filename)
        elif save_path:  # Assumes it's a full file path
            filepath = save_path
        else:
            # Default to current working directory
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = f"history_log_{timestamp}.txt"
            filepath = os.path.join(os.getcwd(), filename)

        try:
            with open(filepath, 'w') as f:
                for rec in filtered_records:
                    timestamp_str = rec['timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    f.write(f"Timestamp: {timestamp_str}\n")
                    f.write(f"Level:     {rec.get('level', 'N/A')}\n")
                    f.write(f"Message:   {rec['description']}\n")
                    f.write("-" * 50 + "\n")
            print(f"Records successfully saved to {filepath}")
            return filepath
        except IOError as e:
            print(f"Error saving records to {filepath}: {e}")
            return ""

    def clear_records(self) -> None:
        """
        Removes all records from the history.
        """
        self.records = []
        # Optionally, add a record indicating the history was cleared,
        # though this might be counter-intuitive if the goal is a complete wipe.
        # self.add_record("History cleared.", level="SYSTEM")

    def __repr__(self) -> str:
        """
        Returns a string representation for debugging.
        """
        return f"InfoManagementSystem(records_count={len(self.records)})"

    def __str__(self) -> str:
        """
        Returns a user-friendly string representation of all records.
        """
        if not self.records:
            return "No records in history."

        report_lines = [f"History Log ({len(self.records)} records):"]
        for rec in self.records:
            timestamp_str = rec['timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            report_lines.append(f"  [{rec.get('level', 'N/A')}] {timestamp_str}: {rec['description']}")
        return "\n".join(report_lines)

