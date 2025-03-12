# History log system for Data Processing and Analysis
# Shenyao Jin, shenyaojin@mines.edu
# 03/09/2025

import datetime


class InfoManagementSystem:
    """
    A simplified information management class that stores records with
    automatic timestamps. No metadata is maintained here.

    Attributes:
    -----------
    records : list of dict
        A list of record dictionaries, each containing a 'description'
        and a 'timestamp' field.
    """

    def __init__(self):
        """
        Initializes the system with an empty list of records.
        """
        self.records = []

    def add_record(self, description):
        """
        Creates a new record with a given description and the current
        timestamp, then appends it to records.

        Parameters
        ----------
        description : str
            The description or information to be stored in the record.
        """
        new_record = {
            'description': description,
            'timestamp': datetime.datetime.now()
        }
        self.records.append(new_record)

    def get_records(self, filter_fn=None):
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
            return self.records
        else:
            return [rec for rec in self.records if filter_fn(rec)]

    def print_records(self, filter_fn=None):
        """
        Prints each record, optionally filtered by a user-defined function.

        Parameters
        ----------
        filter_fn : callable, optional
            A function that takes a single record (dict) and returns a
            boolean. If provided, only records for which filter_fn(record)
            is True will be printed. Defaults to None (no filtering).
        """
        filtered_records = self.get_records(filter_fn)
        for rec in filtered_records:
            print(f"Description: {rec['description']}, Timestamp: {rec['timestamp']}")

    def __repr__(self):
        """
        Returns a string representation for debugging.
        """
        return f"InfoManagementSystem(records_count={len(self.records)})"