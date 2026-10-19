import numpy as np
import pandas as pd
import pytest

from universal import algos

EPS = 1e-10


@pytest.mark.parametrize(
    "algo_class",
    [
        algos.CRP,
        algos.RMR,
        algos.OLMAR,
        algos.PAMR,
        algos.AdaHedge,
    ],
)
def test_bias(algo_class, S):
    """Test forward bias of algo. Test on a portion of given data set, then add several
    data points and see if weights has changed.
    """
    m = 10
    B1 = algo_class().run(S.iloc[:-m]).B
    B2 = algo_class().run(S).B

    assert (B1 == B2.iloc[:-m]).all().all()


# BAH
def test_bah(S):
    """There should be no fees for BAH."""
    FEE = 0.01
    result = algos.BAH().run(S)
    wealth_no_fees = result.total_wealth
    result.fee = FEE
    wealth_with_fees = result.total_wealth

    assert abs(wealth_no_fees - wealth_with_fees) < EPS


# CRP
def test_crp(S):
    """Make sure that equity of a portfolio [1,0,...,0] with NaN values
    is the same as asset itself."""
    b = [1.0] + [0.0] * (len(S.columns) - 1)
    result = algos.CRP(b).run(S)

    assert abs(result.total_wealth - S[S.columns[0]].iloc[-1]) < EPS


def test_tco1(S):
    """Zero turnover with extremely high fees."""
    result = algos.TCO1(eta=1, trx_fee_pct=1e6).run(S)
    assert abs(result.turnover) < 1e-8


# OLMAR
@pytest.mark.parametrize("ma_type", ["SMA", "EMA"])
def test_olmar_min_window(S, ma_type):
    """Window of 2 is the smallest that runs; 1 is rejected with a matching message."""
    result = algos.OLMAR(window=2, ma_type=ma_type).run(S)
    assert result.B.notnull().all().all()

    with pytest.raises(ValueError, match=">=2"):
        algos.OLMAR(window=1, ma_type=ma_type)


def _prices(log_returns):
    return pd.DataFrame(np.exp(np.cumsum(log_returns, axis=0)))


def test_adahedge_follows_consistent_leader():
    """With a leader that never changes, AdaHedge behaves like follow the leader."""
    S = _prices(np.tile([0.01, -0.01, -0.01], (200, 1)))
    B = algos.AdaHedge().run(S).B
    # the first period with returns is row 1, so row 2 must already favour the leader
    assert B.iloc[2, 0] > 0.5
    assert B.iloc[-1, 0] > 1 - 1e-6


def test_adahedge_hedges_against_flipping_leader():
    """The leader flips every period, so following it loses every period. AdaHedge's
    regret to the best asset should grow like sqrt(T), not linearly."""
    T, step = 400, 0.01
    logr = np.zeros((T, 2))
    logr[1] = [-step / 2, 0]
    logr[2::2] = [0, -step]
    logr[3::2] = [-step, 0]
    S = _prices(logr)
    best = np.log(S.iloc[-1]).max()

    regret = best - np.log(algos.AdaHedge().run(S).total_wealth)
    ftl_regret = best - np.log(algos.BestSoFar().run(S).total_wealth)

    assert ftl_regret > T * step / 4
    assert regret < step * np.sqrt(T * np.log(2))
