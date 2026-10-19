import numpy as np
import pandas as pd
import pytest

from universal import algos, tools
from universal.result import ListResult


def _prices(log_returns):
    return pd.DataFrame(np.exp(np.cumsum(log_returns, axis=0)))


def test_follows_consistent_leader():
    """With a leader that never changes, AdaHedge behaves like follow the leader."""
    S = _prices(np.tile([0.01, -0.01, -0.01], (200, 1)))
    B = algos.AdaHedge().run(S).B
    # the first period with returns is row 1, so row 2 must already favour the leader
    assert B.iloc[2, 0] > 0.5
    assert B.iloc[-1, 0] > 1 - 1e-6


def test_hedges_against_flipping_leader():
    """The leader flips every period, so following it loses every period. AdaHedge's
    regret to the best asset stays within the paper's worst-case bound (Corollary 17),
    which grows like sqrt(T), while following the leader loses linearly."""
    T, step, K = 400, 0.01, 2
    logr = np.zeros((T, K))
    logr[1] = [-step / 2, 0]
    logr[2::2] = [0, -step]
    logr[3::2] = [-step, 0]
    S = _prices(logr)
    best = np.log(S.iloc[-1]).max()

    regret = best - np.log(algos.AdaHedge().run(S).total_wealth)
    ftl_regret = best - np.log(algos.BestSoFar().run(S).total_wealth)

    # losses span at most `step` per period
    bound = step * (np.sqrt(T * np.log(K)) + 16 / 3 * np.log(K) + 2)
    assert ftl_regret > T * step / 4
    assert regret < bound


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_single_asset():
    S = tools.random_portfolio(n=100, k=1)
    assert (algos.AdaHedge().run(S).B == 1).all().all()


def test_mixing_strategies(S):
    """Mixing single-asset CRPs is the same as running AdaHedge on the assets."""
    k = S.shape[1]
    experts = [algos.CRP(np.eye(k)[i]).run(S) for i in range(k)]
    mixed = algos.AdaHedge().run(ListResult(experts, list(S.columns)))
    direct = algos.AdaHedge().run(S)

    assert np.allclose(mixed.B, direct.B)
    assert np.isclose(mixed.total_wealth, direct.total_wealth)
