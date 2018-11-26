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
the `ContrastSetLearner` class. Each object requires a pandas DataFrame, 
`frame`, and a feature where rules will be contrasted against. 

```python
from stucco import ContrastSetLearner
learner = ContrastSetLearner(frame, group_feature='color')
```

[1]: https://www.ics.uci.edu/~pazzani/Publications/stucco.pdf
[2]: https://www.anaconda.com/download/
[3]: https://github.com/mwaskom/seaborn-data