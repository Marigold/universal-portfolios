import numpy as np
import pandas as pd
import pytest

from universal import algos
from universal.algo import Algo


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
    """With frequency > 1, each step gets the return since the previous step."""
    algo = _Recorder(price_type, frequency=3)
    algo.run(S)

    xs = pd.DataFrame([x for x, _ in algo.calls])
    # x is labelled with the period it ends at, which is also the end of history
    assert all(x.name == end for x, end in algo.calls)
    assert list(xs.index) == list(S.index[::3])
    assert np.allclose(xs, Algo._convert_prices(S.loc[xs.index], price_type))
