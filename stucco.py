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


class ContrastSetLearner:
    """
    Executes a data mining algorithm known as contrast-set learning. This
    algorithm is designed to learn association rules that have a statistical
    significant presence in one group over another. In doing so, this learning
    algorithm enables identification of potential indicators that describe
    differences across groups, for example: size = small versus size = large.

    Args:
        frame (DataFrame): pandas DataFrame.
        group_feature (str): feature name to drive contrast-set learning.
        num_parts (int): number of partitions floats shall be split into.
        max_unique_reals (int): number of unique reals to justify partitioning.
        max_rows (int): maximum number of DataFrame records to process.

    Raises:
        ValueError: if `group_feature` does not exist or `num_parts` < 1.
    """
    def __init__(self, frame, group_feature, num_parts=3, max_unique_reals=15,
                 max_rows=None):

        if group_feature not in frame:
            raise ValueError('`contrast_feature` must be a valid column name.')

        if num_parts < 1:
            raise ValueError('`num_parts` must be a positive number.')

        # if so-many rows are desired, select those-many rows
        if max_rows:
            frame = pd.DataFrame(frame.iloc[:max_rows])

        # retrieve discrete features, i.e. categorical and boolean, as object
        subset = frame.select_dtypes(['category', 'bool', 'object'])

        # append the feature to its attribute, making it attribute := value
        for col in subset.columns:
            frame[col] = col + ' => ' + frame[col].astype(str)

        # retrieve continuous features, i.e. float and int, as number
        subset = frame.select_dtypes(['number'])

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

        # get the contrast group, remove from frame, and make items as list
        group_values = pd.Series(frame[group_feature], name='group')
        frame.drop(group_feature, axis=1, inplace=True)
        items = pd.Series(frame.apply(lambda x: tuple(x), axis=1), name='items')

        # merge group values and items as DataFrame, and count their frequency
        dummy_frame = pd.concat([group_values, items], axis=1)
        counts = dummy_frame.groupby(list(dummy_frame.columns)).size()

        # data is list containing the items, its group, and count
        self.data = counts.reset_index(name='count').to_dict(orient='records')
        self.group = list(group_values.unique())
