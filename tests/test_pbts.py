import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import LedoitWolf

from universal import tools
from universal.algos import BAH, PBTS
from universal.algos.pbts import long_only_min_variance


def closed_form_delta(R, mv, benchmark):
    """Eq. (9) and (10) worked out: a period is a success exactly when MV beat the
    benchmark in it, so delta = (1 + wins) / (2 + wins + losses)."""
    edge = R @ (mv - benchmark)
    wins, losses = int(np.sum(edge > 1e-12)), int(np.sum(edge < -1e-12))
    return (1 + wins) / (2 + wins + losses)


@pytest.fixture
def window():
    rng = np.random.default_rng(1)
    R = 1 + rng.normal(0, 0.01, size=(50, 4))
    mv = np.array([0.4, 0.3, 0.2, 0.1])
    benchmark = np.ones(4) / 4
    return R, mv, benchmark


def test_sampled_delta_cancels_out(window):
    """The Bernoulli tests reduce to counting the periods MV won, whatever the samples."""
    R, mv, benchmark = window
    expected = closed_form_delta(R, mv, benchmark)
    for seed in range(5):
        algo = PBTS(seed=seed)
        algo.init_step(pd.DataFrame(R))
        assert algo.blending_coefficient(R, mv, benchmark) == pytest.approx(expected)


def test_delta_hand_calculation():
    """Two assets, MV all in the first. It wins, wins, loses: delta = (1 + 2) / (2 + 3)."""
    R = np.array([[1.02, 1.00], [1.01, 0.99], [0.98, 1.03]])
    mv, benchmark = np.array([1.0, 0.0]), np.array([0.5, 0.5])
    algo = PBTS(tau=3)
    algo.init_step(pd.DataFrame(R))
    assert algo.blending_coefficient(R, mv, benchmark) == pytest.approx(3 / 5)


@pytest.mark.parametrize("mv_always_wins, expected", [(True, 11 / 12), (False, 1 / 12)])
def test_delta_limits(mv_always_wins, expected):
    """MV winning (losing) every one of tau periods gives (tau + 1) / (tau + 2) (1 / (tau + 2))."""
    tau = 10
    R = np.tile([1.01, 0.99] if mv_always_wins else [0.99, 1.01], (tau, 1))
    mv, benchmark = np.array([1.0, 0.0]), np.array([0.5, 0.5])
    algo = PBTS(tau=tau)
    algo.init_step(pd.DataFrame(R))
    assert algo.blending_coefficient(R, mv, benchmark) == pytest.approx(expected)


def test_ties_are_not_counted():
    """Footnote 1: when the blends earn the same, a and b stay put. Equal returns tie every period."""
    R = np.ones((6, 2)) * 1.01
    algo = PBTS(tau=6)
    algo.init_step(pd.DataFrame(R))
    delta = algo.blending_coefficient(R, np.array([1.0, 0.0]), np.array([0.5, 0.5]))
    assert delta == pytest.approx(0.5)


def test_long_only_min_variance_matches_hand_calculation():
    """Two uncorrelated assets: weights proportional to 1 / variance."""
    sigma = np.diag([1.0, 4.0])
    assert long_only_min_variance(sigma) == pytest.approx([0.8, 0.2], abs=1e-6)
    # a negatively optimal weight is clipped to zero by the long-only constraint
    sigma = np.array([[1.0, 1.9], [1.9, 4.0]])
    w = long_only_min_variance(sigma)
    assert w == pytest.approx([1.0, 0.0], abs=1e-6)


def transcribe(S, benchmark, tau):
    """Independent loop over the paper's Algorithm 1 using the closed form of eq. (9)-(10)."""
    X = (S / S.shift(1)).fillna(1.0).to_numpy()
    T, n = X.shape
    B = np.zeros((T, n))
    B[0] = np.ones(n) / n
    vw = np.ones(n) / n
    window = []
    for t in range(T - 1):
        vw = vw * X[t] / (vw @ X[t])
        window = (window + [X[t]])[-tau:]
        bench = np.ones(n) / n if benchmark == "EW" else vw
        if len(window) < tau:
            B[t + 1] = bench
            continue
        R = np.array(window)
        mv = long_only_min_variance(np.cov(R - 1, rowvar=False, bias=True))
        delta = closed_form_delta(R, mv, bench)
        B[t + 1] = delta * mv + (1 - delta) * bench
    return B


@pytest.mark.parametrize("benchmark", ["EW", "VW"])
def test_matches_transcription_of_algorithm_1(benchmark):
    S = tools.random_portfolio(n=200, k=4, mu=0.0, sd=0.01)
    B = PBTS(benchmark=benchmark, tau=30).run(S).B.to_numpy()
    assert B == pytest.approx(transcribe(S, benchmark, 30), abs=1e-6)


def test_holds_benchmark_until_tau_periods():
    S = tools.random_portfolio(n=80, k=3, mu=0.0, sd=0.01)
    tau = 40
    ew = PBTS(benchmark="EW", tau=tau).run(S).B
    assert ew.iloc[:tau].to_numpy() == pytest.approx(np.ones((tau, 3)) / 3)
    assert not np.allclose(ew.iloc[tau + 5], 1 / 3)
    # VW is buy and hold from uniform weights
    vw = PBTS(benchmark="VW", tau=tau).run(S).B
    assert vw.iloc[:tau].to_numpy() == pytest.approx(
        BAH().run(S).B.iloc[:tau].to_numpy()
    )


def test_uses_the_latest_return():
    """Planted one-period-lag check: the weights for t + 1 must use the return of period t."""
    S = tools.random_portfolio(n=60, k=3, mu=0.0, sd=0.01)
    B = PBTS(tau=20).run(S).B.to_numpy()
    expected = transcribe(S, "EW", 20)
    lagged = np.vstack([expected[:1], expected[:-1]])
    assert B == pytest.approx(expected, abs=1e-6)
    assert not np.allclose(B[25:], lagged[25:], atol=1e-6)


def test_single_asset():
    S = tools.random_portfolio(n=50, k=1, mu=0.0, sd=0.01)
    assert PBTS(tau=10).run(S).B.to_numpy() == pytest.approx(np.ones((50, 1)))


def test_duplicated_asset_is_finite():
    S = tools.random_portfolio(n=80, k=2, mu=0.0, sd=0.01)
    S["dup"] = S.iloc[:, 0]
    B = PBTS(tau=30).run(S).B.to_numpy()
    assert np.all(np.isfinite(B)) and B.min() >= -1e-9
    assert B.sum(axis=1) == pytest.approx(1)


def test_asset_listed_late():
    S = tools.random_portfolio(n=80, k=3, mu=0.0, sd=0.01)
    S.iloc[:20, 2] = np.nan
    B = PBTS(tau=30).run(S).B.to_numpy()
    assert np.all(np.isfinite(B)) and B.sum(axis=1) == pytest.approx(1)


def test_other_covariance_estimators():
    S = tools.random_portfolio(n=80, k=3, mu=0.0, sd=0.01)
    for estimator in ("ledoit-wolf", "oas", LedoitWolf()):
        B = PBTS(tau=30, cov_estimator=estimator).run(S).B.to_numpy()
        assert np.all(np.isfinite(B)) and B.sum(axis=1) == pytest.approx(1)


def test_rejects_bad_parameters():
    with pytest.raises(ValueError, match="benchmark"):
        PBTS(benchmark="MV")
    with pytest.raises(ValueError, match="tau"):
        PBTS(tau=1)
    with pytest.raises(ValueError, match="cov_estimator"):
        PBTS(cov_estimator="factor")
