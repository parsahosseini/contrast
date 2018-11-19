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
