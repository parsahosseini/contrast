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

To execute contrast-set learning, using the STUCCO algorithm, we can leverage
the `ContrastSetLearner` class. Each such object requires a pandas DataFrame, 
`frame`, and a feature where rules will be contrasted against.

```python
# a skeleton object; no contrast-set analysis has taken place.
from stucco import ContrastSetLearner
learner = ContrastSetLearner(frame, group_feature='color')
```

Our feature may have many states, i.e. `('color' == 'D', 'color' == 'E'`. As
a result, the goal of contrast-set learning is to derive which association
rules are enriched in one group-state over another. To make
this happen, we have to quantify rule abundance. 

*Recommendations*

We recommend experimenting with `learner.learn()` parameter arguments, namely
`max_length`. This parameter references the maximum length of an item-set
following derivation of its canonical combinations. In other words, the smaller
this length, the fewer rules will be learned. On the other hand, the higher 
this value, the longer the rules shall be; we recommend executing with
at least `max_length = 2`, and adjusting accordingly.

```python
# process how many times each rule is found in each group-state.
learner.learn()
```

Extracting association rules enriched across an exclusive set of groups
is the next phase in contrast-set learning. Here, statistical metrics are 
leveraged capable of assessing rule abundance in one group over another.
Such metrics include:

* Support `p(A, B)`
* Lift `p(A, B) / p(A) * p(B)`
* Confidence `max((p(A, B) / p(A)), (p(A, B) / p(B)))`

Such metrics are leveraged in `learner.score()`, with parameters to accept
to accept argument values for the metrics. The end result, `output`, is a
pandas `DataFrame` that references the rule, its group, and its lift score.

```python
output = learner.score(min_lift=3)
```

*Recommendations*

Increasing `learner.score()` parameter arguments renders scoring to be more
stringent, and thus returns fewer intelligible rules. We recommend 
experimenting with such parameter-values, namely `min_lift` and `min_support`.

[1]: https://www.ics.uci.edu/~pazzani/Publications/stucco.pdf
[2]: https://www.anaconda.com/download/
[3]: https://github.com/mwaskom/seaborn-data