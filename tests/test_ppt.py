import numpy as np
import pandas as pd
import pytest

from universal import algos, tools


def test_hand_computed_steps():
    """With two assets the centered prediction points along (1, -1) / sqrt(2), so a step
    of eps = 0.1 * sqrt(2) moves 0.1 of weight to the asset with the higher predicted
    price relative, or nothing on a tie.

    window = 2, so step 1 (two prices seen) uses the last price relative and later
    steps use max(last two prices) / last price:
    - step 1: relatives (1, 2), B rises -> B[2] = (0.4, 0.6)
    - step 2: peaks (1, 4/4), tie -> B[3] = B[2]
    - step 3: peaks (1, 4/2), B is below its peak -> B[4] = (0.3, 0.7)
    - step 4: peaks (1, 2/1) -> B[5] = (0.2, 0.8)
    """
    S = pd.DataFrame({"A": [1.0] * 6, "B": [1.0, 2.0, 4.0, 2.0, 1.0, 1.0]})
    B = algos.PPT(window=2, eps=0.1 * np.sqrt(2)).run(S).B

    expected = [[0.5, 0.5], [0.5, 0.5], [0.4, 0.6], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8]]
    assert np.allclose(B, expected)


def test_step_has_length_eps():
    """Away from the simplex boundary the update is b + eps * c / |c|, where c is the
    predicted price relative minus its mean: weights keep summing to one and move by
    exactly eps, or not at all when every asset has the same prediction."""
    np.random.seed(1)
    S = tools.random_portfolio(n=50, k=4, sd=0.02)
    eps = 1e-3
    B = algos.PPT(window=5, eps=eps).run(S).B

    moves = np.linalg.norm(np.diff(B.values, axis=0), axis=1)
    assert np.allclose(B.sum(axis=1), 1)
    assert np.allclose(moves[0], 0)  # no step has been taken before row 2
    assert np.allclose(moves[1:], eps)


def test_large_eps_goes_all_in_on_furthest_below_peak():
    """For eps -> inf the projection lands on the vertex of the asset with the highest
    predicted price relative: once the window is full, the asset trading furthest
    below its peak over the last `window` prices."""
    np.random.seed(2)
    S = tools.random_portfolio(n=60, k=4, sd=0.02)
    window = 5
    B = algos.PPT(window=window, eps=1e4).run(S).B

    peak_ratio = S.rolling(window).max() / S
    for t in range(window, len(S) - 1):
        assert B.iloc[t + 1].idxmax() == peak_ratio.iloc[t].idxmax()
        assert np.isclose(B.iloc[t + 1].max(), 1)


def test_warm_up_follows_last_price_relative():
    """Until a full window of prices is available, the prediction is the last price
    relative, so a large step goes all in on the last period's best asset."""
    np.random.seed(3)
    S = tools.random_portfolio(n=10, k=3, sd=0.02)
    window = 5
    B = algos.PPT(window=window, eps=1e4).run(S).B

    best = (S / S.shift(1)).iloc[1:].idxmax(axis=1)
    for t in range(1, window):
        assert B.iloc[t + 1][best.loc[t]] == pytest.approx(1)


def test_window_one_is_uniform_crp(S):
    """With window = 1 the peak is the current price, every prediction is 1 and PPT
    never moves away from uniform weights."""
    result = algos.PPT(window=1).run(S)

    assert np.allclose(result.B, 1 / S.shape[1])
    assert np.isclose(result.total_wealth, algos.CRP().run(S).total_wealth)


def test_small_eps_stays_near_uniform(S):
    B = algos.PPT(eps=1e-9).run(S).B
    assert np.allclose(B, 1 / S.shape[1], atol=1e-7)


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize(
    "S",
    [tools.random_portfolio(n=50, k=1), pd.DataFrame(np.ones((20, 3)))],
    ids=["single_asset", "constant_prices"],
)
def test_no_step_possible(S):
    """A single asset, or constant prices that predict no change for any asset, leave
    nothing to step towards, so the weights stay uniform without warnings."""
    assert np.allclose(algos.PPT().run(S).B, 1 / S.shape[1])


def test_late_listed_asset():
    """A missing price means no change: an asset listed late behaves as if it had
    traded at its listing price before, and its NaNs never reach the weights."""
    np.random.seed(4)
    S = tools.random_portfolio(n=80, k=3, sd=0.02)
    S_late = S.copy()
    S_late.iloc[:30, 2] = np.nan
    S_flat = S.copy()
    S_flat.iloc[:30, 2] = S.iloc[30, 2]

    B_late = algos.PPT().run(S_late).B
    B_flat = algos.PPT().run(S_flat).B

    assert np.isfinite(B_late).all().all()
    assert np.allclose(B_late, B_flat)


def test_duplicated_asset():
    """Identical assets get identical predictions and so identical weights."""
    np.random.seed(5)
    S = tools.random_portfolio(n=80, k=3, sd=0.02)
    S["S2_copy"] = S["S2"]
    B = algos.PPT().run(S).B

    assert np.allclose(B["S2"], B["S2_copy"])
    assert np.allclose(B.sum(axis=1), 1)


@pytest.mark.parametrize("kwargs", [{"window": 0}, {"eps": 0}, {"eps": -1}])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        algos.PPT(**kwargs)
