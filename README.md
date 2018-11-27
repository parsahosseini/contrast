contrast
========

contrast is a Python implementation of the STUCCO algorithm ([URL][1]) that 
allows the ability to learn association rules with significant 
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

A pandas `DataFrame` is needed to drive contrast-set learning. For 
demonstration purposes, we shall leverage existing [seaborn datasets][3].

```python
# read-in data and produce a DataFrame
from seaborn import load_dataset
frame = load_dataset('diamonds')
```

To execute the algorithm, we leverage the `ContrastSetLearner` class. A
minimally working instance needs two arguments:

* A pandas `DataFrame`
* A feature name, `group_feature`

```python
# a skeleton object; no contrast-set analysis has taken place.
from stucco import ContrastSetLearner
learner = ContrastSetLearner(frame, group_feature='color')
```

A feature can have many states, i.e. `color = {'D', 'E', 'F', 'G'}`. Thus, 
the goal of contrast-set learning is to gauge what rules are enriched across 
different group states. To make this happen, we leverage `learner.learn()` to 
enumerate rule abundance.

*Considerations*

We recommend tweaking `learner.learn()` parameter values, namely
`max_length`. This parameter dictates the maximum rule length 
following derivation of its canonical combinations. For example: suppose we 
have the rule `['a = 1', 'b = 2', 'c = 3']`. If `max_length=2`, all 
rule combinations, of length `max_length`, are produced. 
Due to this combinatorial function, an important consideration must be made:

* Large `max_length`: increases runtime, possibility of intelligible rules.
* Small `max_length`: quick runtime, few intelligible rules.

```python
# derive 3-length combinations of a rule and enumerate their abundance.
learner.learn(max_length=3)
```

To determine if a rule, `A`, is enriched in a desired groups' state, `B`, 
rule counts get modeled as a 2 x 2 contingency matrix, `m`. We denote "not" 
symbol as `~`:

m     | B       |    ~B     |
:---: | :---:   |   :---:   |
A     | p(A, B) | p(A, ~B)  |
~A    | p(~A, B)| p(~A, ~B) |

Given `m`, we can quantify rule abundance using several statistical metrics:

* Support: `p(A, B)`
* Lift: `p(A, B) / p(A) * p(B)`
* Confidence: `max(p(A, B) / p(A), p(A, B) / p(B))`

These metrics are invoked in `learner.score()`. Their collective outputs must 
exceed user-provided thresholds in order to be deemed enriched in a state.
Following completion of quantification, `output`, a `DataFrame` that references 
the rule, its group state, and its satisfactory lift score, is returned.

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