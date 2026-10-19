import numpy as np
import pytest

from universal import algos

EPS = 1e-10

ALL_ALGOS = [getattr(algos, name) for name in algos.__all__]

# Optimal in hindsight by definition, so their weights depend on future prices.
HINDSIGHT = {algos.BCRP, algos.BestMarkowitz}

# Compute all weights at once in weights() and have no step().
NO_STEP = {algos.BAH, algos.Anticor}

# Long-short and leveraged by design.
LEVERAGED = {algos.Kelly}


def _run(algo_class, S):
    # UP and BestSoFar draw random numbers; seed so that two runs are comparable
    np.random.seed(0)
    return algo_class().run(S).B


@pytest.mark.parametrize(
    "algo_class", [a for a in ALL_ALGOS if a not in HINDSIGHT], ids=lambda a: a.__name__
)
def test_bias(algo_class, S):
    """Test forward bias of algo. Test on a portion of given data set, then add several
    data points and see if weights has changed.
    """
    m = 10
    B1 = _run(algo_class, S.iloc[:-m])
    B2 = _run(algo_class, S)

    assert np.allclose(B1, B2.iloc[:-m])


@pytest.mark.parametrize(
    "algo_class",
    [a for a in ALL_ALGOS if a not in HINDSIGHT | NO_STEP],
    ids=lambda a: a.__name__,
)
def test_next_weights_matches_run(algo_class, S):
    """next_weights on prices up to the second-to-last period gives the weights run
    holds in the last period, also for algorithms that keep state between steps."""
    B = _run(algo_class, S)
    np.random.seed(0)
    b = algo_class().next_weights(S.iloc[:-1], B.iloc[-2])

    assert np.allclose(b, B.iloc[-1])


@pytest.mark.parametrize(
    "algo_class",
    [a for a in ALL_ALGOS if a not in LEVERAGED],
    ids=lambda a: a.__name__,
)
def test_long_only_weights(algo_class, S):
    """Weights are finite, non-negative and sum to at most one; anything below one is
    held in cash."""
    B = _run(algo_class, S)

    assert B.shape == S.shape
    assert np.isfinite(B).all().all()
    assert (B >= -EPS).all().all()
    assert (B.sum(axis=1) <= 1 + 1e-6).all()


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


# SoftBayes
@pytest.mark.parametrize("eta, benchmark", [(1.0, algos.BAH), (0.0, algos.CRP)])
def test_soft_bayes_limits(eta, benchmark, S):
    """eta=1 lets weights drift like BAH, eta=0 keeps them uniform like CRP."""
    result = algos.SoftBayes(eta=eta).run(S)

    assert abs(result.total_wealth - benchmark().run(S).total_wealth) < EPS
