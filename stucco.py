import json
import numpy as np
import pandas as pd

from bisect import bisect_left, bisect_right, bisect
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


def frame_to_items(frame, max_num_items=None, **kwargs):
    """
    Parses DataFrame records as {column} := {value} format, where {column} is
    the DataFrame feature, and {value} is its value. For continuous features,
    (i.e. floats and ints), {value} is the upper- and lower-limits this numeric
    value falls in-between. Applying such logic, for both continuous and
    discrete features, helps in modeling feature-specific context and better
    understanding of enriched association rules.

    Args:
        frame (DataFrame): pandas DataFrame.
        group_feature (str): feature in `frame` to contrast rules against.
        max_num_items (int): maximum number of items, or records, to parse.
        kwargs (dict): Keyword arguments accepted by `get_frame_metadata`)

    Raises:
        ValueError: if `group_feature` in not a valid DataFrame column name.

    Yields:
        (list, str): items and its corresponding group feature (optional)
    """

    # derive the metadata for the DataFrame
    metadata = get_frame_metadata(frame, **kwargs)

    # each DataFrame row has a list of items and an optional group
    for row_num, row in frame.iterrows():
        items = []

        # get feature and its value, and get data-type so right item is fetched
        for col_name, value in list(row.items()):
            data_type = np.dtype(metadata[col_name]['data_type'])

            # if object, simply define the feature as the respective value
            if data_type == np.object:
                item = '{} := {}'.format(col_name, value)

            # if real, its values could be a list or matrix
            elif data_type == np.float or np.int:
                values = metadata[col_name]['values']

                # if list, define the feature as the respective value
                if np.ndim(values) == 1:
                    item = '{} := {}'.format(col_name, value)

                # if array, define feature as the bounds the value falls in
                else:
                    split = filter(lambda x: x[0] <= value <= x[-1], values)
                    value = np.ravel(list(split))
                    item = '{} := ({}...{})'.format(col_name, *value)

            else:
                msg = '{} must be numpy object, float, or int'.format(col_name)
                raise TypeError(msg)

            items.append(item)

        # break generation of items if so-many items have been generated
        if max_num_items == row_num:
            break

        yield items


def get_frame_metadata(frame, num_splits=3, max_unique_reals=15, **kwargs):
    """
    Derive the metadata for a pandas DataFrame. Metadata is defined as the
    data-type and data-range of a feature.

    Args:
        num_splits (int): number of partitions to split continuous features.
        max_unique_reals (int): number of unique reals before deemed float.

    Keyword Args:
        out (file): output JSON filename to write metadata object to.

    Raises:
        ValueError: if `num_splits` is less than 1 or None

    Returns:
          dict: feature-metadata pairing, otherwise known as metadata.
    """

    # ensure the number of bins to split floats into is a valid integer
    if num_splits is None or num_splits < 1:
        raise ValueError('`num_splits` must be a positive integer.')

    # store metadata given a DataFrame
    metadata = {}

    # fetch discrete features data-types, get metadata, and save to dict
    subset = frame.select_dtypes(['category', 'bool', 'object'])
    frame[subset.columns] = frame[subset.columns].astype(object)

    for col_name in subset:
        values = subset[col_name].unique()
        record = {'values': values.tolist(), 'data_type': str(values.dtype)}
        metadata[col_name] = record

    # fetch real-valued data-types, get metadata, and save to dict
    subset = frame.select_dtypes(['int', 'float'])
    for col_name in subset:
        values = subset[col_name].sort_values().unique()

        # some features are continuous, so partition and fetch boundary ranges
        if len(values) > max_unique_reals:
            splits = np.array_split(values, num_splits)
            values = np.asarray(list(map(lambda x: list(x[[0, -1]]), splits)))
        record = {'values': values.tolist(), 'data_type': str(values.dtype)}
        metadata[col_name] = record

    print(frame.dtypes)
    # persist metadata object as JSON
    out_file = kwargs.get('out')
    if out_file:
        json.dump(metadata, open(out_file, 'w'), indent=4)
    return metadata


def stucco(items_or_iterator, group_feature, max_rule_length=2):
    for items in items_or_iterator:
        pass

        # check to see if `group_feature` exists in `items`; if its count != 1,
        # then break.

        # have dict; K => comb; V => {group label: its count}


class ContrastSetLearner:
    def __init__(self, frame, contrast_feature, num_parts=3,
                 max_unique_reals=15):

        if contrast_feature not in frame:
            raise ValueError('`contrast_feature` must be a valid column name.')

        if num_parts < 1:
            raise ValueError('`num_parts` must be a positive number.')

        # cast discrete features, i.e. categorical and boolean, as object
        subset = frame.select_dtypes(['category', 'bool', 'object'])
        # frame[subset.columns] = frame[subset.columns].astype(np.object)

        # append the feature to its attribute, making it attribute := value
        for col in subset.columns:
            frame[col] = col + ' => ' + frame[col].astype(str)

        # cast continuous features, i.e. float and int, as number
        subset = frame.select_dtypes(['number'])
        # frame[subset.columns] = frame[subset.columns].astype(np.number)

        # repeat the appending process above, but for real-values
        for col in subset.columns:
            series = frame[col]

            # if numeric feature has many unique values, partition into chunks
            if len(set(series)) > max_unique_reals:
                arr = series.sort_values().unique()
                parts = np.array_split(arr, num_parts)

                # partitions have (lower, upper) value; use lower to get index
                values = list(map(lambda x: (x[0], x[-1]), parts))
                lwr = list(map(lambda x: x[0], values))

                # determine which (lower, upper) range this value falls into
                series = series.apply(lambda x: values[bisect_right(lwr, x)-1])
                frame[col] = col + ' => ' + series.astype(str)

            # if numeric feature has few unique values, append it like object
            else:
                frame[col] = col + ' => ' + frame[col].astype(str)
        self.frame = frame
