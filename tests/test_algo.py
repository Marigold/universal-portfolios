import numpy as np
import pandas as pd
import pytest

from universal import algos
from universal.algo import Algo

# Optimal in hindsight by definition, so there is no "next" weight to compute.
HINDSIGHT = {algos.BCRP, algos.BestMarkowitz}
# Compute all weights at once in weights() and have no step().
NO_STEP = {algos.BAH, algos.Anticor}


@pytest.mark.parametrize(
    "algo_class",
    [
        getattr(algos, name)
        for name in algos.__all__
        if getattr(algos, name) not in HINDSIGHT | NO_STEP
    ],
    ids=lambda a: a.__name__,
)
def test_next_weights_matches_run(algo_class, S):
    """next_weights on prices up to the second-to-last period gives the weights run
    holds in the last period, also for algorithms that keep state between steps."""
    np.random.seed(0)
    B = algo_class().run(S).B
    np.random.seed(0)
    b = algo_class().next_weights(S.iloc[:-1], B.iloc[-2])

    assert np.allclose(b, B.iloc[-1])


def test_n_jobs_removed(S):
    with pytest.raises(ValueError, match="n_jobs=1"):
        algos.CRP().run(S, n_jobs=2)


class _Recorder(Algo):
    """Records the x and history passed to step."""

    def __init__(self, price_type, frequency):
        super().__init__(frequency=frequency)
        self.PRICE_TYPE = price_type
        self.calls = []

    def init_weights(self, columns):
        return np.ones(len(columns)) / len(columns)

    def step(self, x, last_b, history):
        self.calls.append((x, history.index[-1]))
        return last_b


@pytest.mark.parametrize("price_type", ["ratio", "log"])
def test_frequency_accumulates_skipped_periods(price_type, S):
    """With frequency > 1, step still sees the return over every period."""
    algo = _Recorder(price_type, frequency=3)
    algo.run(S)
    X = Algo._convert_prices(S, price_type)

    xs = pd.DataFrame([x for x, _ in algo.calls])
    # x is labelled with the period it ends at, which is also the end of history
    assert all(x.name == end for x, end in algo.calls)
    assert list(xs.index) == list(X.index[::3])

    total = xs.prod() if price_type == "ratio" else xs.sum()
    seen = X.loc[: xs.index[-1]]
    expected = seen.prod() if price_type == "ratio" else seen.sum()
    assert np.allclose(total, expected)
