#!/usr/bin/env python3
"""From File Module"""

import pandas as pd

def from_file(filename, delimiter):
    """Loads data from a file as a pd.DataFrame:

        filename is the file to load from
        delimiter is the column separator
        Returns: the loaded pd.DataFrame"""

    df = pd.read_csv(filename, sep=delimiter)

    return df
