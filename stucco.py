import json
import numpy as np
import pandas as pd

from itertools import combinations


def canonical_combination(items, max_length=None):
    """
    Generates all variable-length combinations, given a list.

    Args:
        items (list): collection of objects.
        max_length (int): maximum combination subset length.

    Examples:
        >>> list(canonical_combination([1, 2, 3]))
        [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]

        >>> list(canonical_combination(['a', 'b', 'c'], max_length=2))
        [('a',), ('b',), ('c',), ('a', 'b'), ('a', 'c'), ('b', 'c')]

    Yields:
        list: a variable-length combination
    """
    for length in range(1, len(items) + 1):
        for subset in combinations(items, length):
            yield subset
        if length == max_length:
            break


def get_frame_metadata(frame, num_splits=3, max_unique_reals=15, out=None):
    """
    Derive the metadata for a pandas DataFrame. Metadata is defined as the
    data-type and data-range of a feature.
    """

    # ensure the number of bins to split floats into is a valid integer
    if num_splits is None or num_splits < 1:
        raise ValueError('`num_splits` must be a positive integer.')

    # store metadata given a DataFrame
    metadata = {}

    # handle categorical data-points, i.e. category, boolean, and strings
    subset = frame.select_dtypes(['category', 'bool', 'object'])
    for col_name in subset:
        values = subset[col_name].unique()
        record = {'items': values.tolist(), 'data_type': str(values.dtype)}
        metadata[col_name] = record

    # handle real-data; too many points makes them continuous in-range
    subset = frame.select_dtypes(['int', 'float'])
    for col_name in subset:
        values = subset[col_name].unique()
        if len(values) > max_unique_reals:
            splits = np.array_split(values, num_splits)
            values = np.asarray(list(map(lambda x: list(x[[0, -1]]), splits)))
        record = {'items': values.tolist(), 'data_type': str(values.dtype)}
        metadata[col_name] = record

    if out:
        json.dump(metadata, open(out, 'w'), indent=4)
