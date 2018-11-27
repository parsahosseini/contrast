contrast
========

contrast is a Python implementation of the STUCCO algorithm ([URL][1]) that 
allows the ability to learn association rules that exhibit significant 
enrichment in one group over another. As a result, the produced association 
rules can help "describe" indicators aligned to a group of interest.

Dependencies
------------
- Python 3.7+
- numpy (1.15+)
- pandas (0.23+)

Such libraries can also be installed using the [Anaconda Python installer][2]. 

Examples
--------

A pandas DataFrame is needed to drive contrast-set learning. For demonstration
purposes, we shall leverage existing [seaborn datasets][3].

```python
# read-in data and produce a DataFrame
from seaborn import load_dataset
frame = load_dataset('diamonds')
```

To execute STUCCO contrast-set learning, we can leverage
the `ContrastSetLearner` class. Each such object requires a pandas `DataFrame` 
and a column name or feature, `group_feature`, where rules will be contrasted against.

```python
# a skeleton object; no contrast-set analysis has taken place.
from stucco import ContrastSetLearner
learner = ContrastSetLearner(frame, group_feature='color')
```

Our feature of interest is merely a column name in our frame, however it
may have many states, i.e. `('color' == 'D', 'color' == 'E')`. The ultimate
goal of contrast-set learning is to gauge what rules are enriched in which 
states. To make this happen, we invoke `learner.learn()` which is capable
of enumerating rule abundance across each state. 

*Considerations*

We recommend experimenting with `learner.learn()` parameter arguments, namely
`max_length`. This parameter is the maximum length of a rule
following derivation of its canonical combinations. For example: suppose we 
have the array `x = ['a', 'b', 'c']`. Setting `max_length=2` would give all
combinations of at least this length, or more formally `['a', 'b', 'c', ('a', 'b'), 
('a', 'c'), ('b', 'c')]`. Thus, a large `max_length` increases runtime but 
produces high-resolution rules. 


```python
# process how many times each rule is found in each group-state.
learner.learn(max_length=3)
```

Extracting association rules enriched across an exclusive set of groups
is the goal of contrast-set learning. To make this possible, rule abundance across groups
is modeled as a 2 x 2 contingency matrix. In this matrix, we model our rule, `A`, and its group 
state, `B`. We denote "not" symbol as `~`:

Matrix| B       |    ~B     |
:---: | :---:   |   :---:   |
A     | p(A, B) | p(A, ~B)  |
~A    | p(~A, B)| p(~A, ~B) |


Given our matrix, we can now represent our statistical metrics:
* Support `p(A, B)`
* Lift `p(A, B) / p(A) * p(B)`
* Confidence `max(p(A, B) / p(A), p(A, B) / p(B))`

Such metrics are leveraged in `learner.score()`, with parameters to accept
to accept argument values for the metrics. The end result, `output`, is a
pandas `DataFrame` that references the rule, its group, and its lift score.

```python
output = learner.score(min_lift=3)
```

*Considerations*

Increasing `learner.score()` parameter arguments renders scoring to be more
stringent, and thus returns fewer intelligible rules. Therefore, we recommend 
experimenting with such parameter-values, namely `min_lift` and `min_support`.

[1]: https://www.ics.uci.edu/~pazzani/Publications/stucco.pdf
[2]: https://www.anaconda.com/download/
[3]: https://github.com/mwaskom/seaborn-data