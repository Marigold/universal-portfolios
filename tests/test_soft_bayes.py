import numpy as np
import pandas as pd
import pytest

from universal import algos

EPS = 1e-10


def _prices(relatives):
    """Prices starting at 1 whose price relatives are the given rows. The framework
    treats the first row of prices as a period with all relatives equal to 1."""
    R = np.asarray(relatives, dtype=float)
    S = np.vstack([np.ones(R.shape[1]), np.cumprod(R, axis=0)])
    return pd.DataFrame(S, columns=[f"S{i}" for i in range(R.shape[1])])


def _log_wealth(algo, S):
    return np.log(algo.run(S, log_progress=False).total_wealth)


def _flipping_leader(T, third=None):
    """The leader flips every period, by a factor of 4. Buy and hold ends where it
    started, while rebalancing to (1/2, 1/2) gains ln(1.25) per period. An optional
    third asset with a constant relative `third` < 2 is never the leader."""
    rows = [[2.0, 0.5], [0.5, 2.0]] * (T // 2)
    if third is not None:
        rows = [r + [third] for r in rows]
    return _prices(rows)


def _theory_eta(T, N, m):
    """eta of Theorem 3: eta_bar = eta / (1 - eta) = sqrt(ln N / (T m))."""
    eta_bar = np.sqrt(np.log(N) / (T * m))
    return eta_bar / (1 + eta_bar)


# limits


def test_eta_one_is_bah(S):
    """With eta = 1 the weights are the Bayesian posterior, i.e. they drift with prices
    like buy and hold."""
    sb = algos.SoftBayes(eta=1.0).run(S)
    bah = algos.BAH().run(S)

    assert np.allclose(sb.B, bah.B)
    assert abs(sb.total_wealth - bah.total_wealth) < EPS


@pytest.mark.parametrize("eta", [0.0, 1e-9])
def test_eta_to_zero_is_uniform_crp(eta, S):
    sb = algos.SoftBayes(eta=eta).run(S)
    crp = algos.CRP().run(S)

    assert np.allclose(sb.B, 1 / S.shape[1], atol=1e-8)
    assert abs(sb.total_wealth - crp.total_wealth) < 1e-8


def test_eta_out_of_range():
    with pytest.raises(ValueError):
        algos.SoftBayes(eta=1.5)


# hand-computed cases


def test_hand_computed_fixed_eta():
    """eq. (7) with eta = 1/2, worked out by hand. Row t of B holds the weights computed
    from relatives up to row t - 1."""
    S = _prices([[2, 1], [1, 3], [1, 1]])
    B = algos.SoftBayes(eta=0.5).run(S).B.values

    # row 0 has relatives 1, which leave the weights unchanged
    assert np.allclose(B[:2], 0.5)
    # x = (2, 1), M = 3/2: b_1 = 1/2 * (1/2 + 1/2 * 2 / (3/2)) = 7/12
    assert np.allclose(B[2], [7 / 12, 5 / 12])
    # x = (1, 3), M = 7/12 + 15/12 = 11/6: b_1 = 7/12 * (1/2 + 1/2 * 6/11) = 119/264
    assert np.allclose(B[3], [119 / 264, 145 / 264])


def test_hand_computed_anytime():
    """eq. (14) with eta_t = sqrt(ln 2 / (4 t)), worked out by hand. Each period is a
    round, including row 0."""
    S = _prices([[2, 1], [1, 1]])
    B = algos.SoftBayes().run(S).B.values

    # round 1 has relatives 1, and the correction mixes uniform weights with themselves
    assert np.allclose(B[1], 0.5)
    # round 2: x = (2, 1), M = 3/2, so 1 - eta + eta * x_1 / M = 1 + eta / 3, then the
    # correction with eta_3 / eta_2 = sqrt(2 / 3) pulls it back towards 1/2
    eta2 = np.sqrt(np.log(2) / 8)
    b1 = np.sqrt(2 / 3) * 0.5 * (1 + eta2 / 3) + (1 - np.sqrt(2 / 3)) * 0.5
    assert np.allclose(B[2], [b1, 1 - b1])


# guarantees


@pytest.mark.parametrize("eta", [0.05, 0.3, 1.0])
@pytest.mark.parametrize("case", ["dominant", "flipping"])
def test_theorem_1_regret_to_single_asset(eta, case):
    """Theorem 1: the log-wealth regret against any single asset is at most
    (1 / eta) ln(1 / w_1^i) = (1 / eta) ln N with uniform prior weights."""
    S = {
        # one asset grows while the others fall, so eta = 1 (BAH) gets close to ln N
        "dominant": _prices([[1.1, 0.9, 0.9]] * 200),
        "flipping": _flipping_leader(400),
    }[case]
    N = S.shape[1]

    asset_log_wealth = np.log(S.iloc[-1] / S.iloc[0])
    regret = asset_log_wealth - _log_wealth(algos.SoftBayes(eta=eta), S)

    assert (regret <= np.log(N) / eta + 1e-9).all()


@pytest.mark.parametrize("third", [None, 0.9], ids=["N=2", "N=3,m=2"])
def test_theorem_3_regret_to_best_crp(third):
    """Theorem 3, with a leader that flips every period: the regret against any CRP a is
    at most (1 / eta_bar) ln N + eta_bar m T + m ln(N / m) + ln N, where m is the number
    of assets that are the best in at least one period. With the theory eta the bound
    grows like sqrt(T), and buy and hold's regret, linear in T, is above it, so the
    bound is not vacuous on this data."""
    T = 1000
    S = _flipping_leader(T, third)
    N = S.shape[1]
    # the first row of relatives is all ones and leaves fixed-eta weights unchanged,
    # so the rounds are the T rows after it
    X = (S / S.shift(1)).iloc[1:].values
    m = int((X == X.max(axis=1, keepdims=True)).any(axis=0).sum())
    eta = _theory_eta(T, N, m)
    eta_bar = eta / (1 - eta)
    bound = np.log(N) / eta_bar + eta_bar * m * T + m * np.log(N / m) + np.log(N)

    # by symmetry the best CRP is (1/2, 1/2, 0); the optimiser should find it too
    a_half = algos.CRP(b=np.array([0.5, 0.5] + ([0.0] if third else [])))
    bcrp = algos.BCRP(no_cash=True)
    sb = _log_wealth(algos.SoftBayes(eta=eta), S)

    for comparator in [a_half, bcrp]:
        assert _log_wealth(comparator, S) - sb <= bound
    assert _log_wealth(a_half, S) - _log_wealth(algos.BAH(), S) > bound


@pytest.mark.parametrize("third", [None, 0.9], ids=["N=2", "N=3"])
def test_theorem_10_anytime_regret_to_best_crp(third):
    """Theorem 10: with eta_t = sqrt(ln N / (2 N t)) and the correction of eq. (14), the
    regret against any CRP over T rounds is at most
    2 sqrt(2 (T + 1) N ln N) + (N / 2 + ln N) ln(T + 1) + ln N, without knowing T."""
    S = _flipping_leader(1000, third)
    # every row of S is a round of the anytime learning rate
    T, N = S.shape
    bound = (
        2 * np.sqrt(2 * (T + 1) * N * np.log(N))
        + (N / 2 + np.log(N)) * np.log(T + 1)
        + np.log(N)
    )

    best = _log_wealth(algos.BCRP(no_cash=True), S)
    sb = _log_wealth(algos.SoftBayes(), S)

    assert best - sb <= bound
    assert best - _log_wealth(algos.BAH(), S) > bound


@pytest.mark.parametrize("eta", [0.01, 0.2, 0.9])
@pytest.mark.parametrize("case", ["random", "flipping"])
def test_weight_change_is_capped(eta, case, S):
    """By dominance (M_t >= b_{t,i} x_{t,i}), one step keeps b_{t+1,i} within
    [(1 - eta) b_{t,i}, (1 - eta) b_{t,i} + eta]: no asset gains more than eta or loses
    more than a fraction eta of its weight, however extreme the returns."""
    if case == "flipping":
        # a factor of 100 per period, where EG's exponential update jumps far more
        S = _prices([[10.0, 0.1], [0.1, 10.0]] * 50)
    B = algos.SoftBayes(eta=eta).run(S).B.values
    prev, nxt = B[:-1], B[1:]

    assert (nxt >= (1 - eta) * prev - EPS).all()
    assert (nxt <= (1 - eta) * prev + eta + EPS).all()


def test_anytime_weights_restart():
    """eq. (16): the correction keeps every weight above w_1 (1 - eta_t / eta_{t-1}), so
    an asset that has done badly for a long time can still recover."""
    S = _prices([[1.0, 0.1]] * 200)
    B = algos.SoftBayes().run(S).B.values

    # row t holds the weights after round t, w_{t+1} >= w_1 (1 - sqrt(t / (t + 1)))
    t = np.arange(1, len(B))
    assert (B[1:, 1] >= 0.5 * (1 - np.sqrt(t / (t + 1))) - EPS).all()


# edge cases


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("eta", [None, 0.1])
def test_single_asset(eta):
    S = _prices([[1.1], [0.9], [1.0]])
    assert (algos.SoftBayes(eta=eta).run(S).B == 1).all().all()


@pytest.mark.parametrize("eta", [None, 0.1])
def test_constant_prices(eta):
    """Nothing to learn from, so the weights stay uniform."""
    S = _prices(np.ones((50, 3)))
    assert np.allclose(algos.SoftBayes(eta=eta).run(S).B, 1 / 3)


@pytest.mark.parametrize("eta", [None, 0.1])
def test_late_listed_asset(eta, S):
    """Before its first price, an asset is treated as having a constant price: it holds
    its share of the weights in cash. Without that, NaNs would spread to every weight."""
    late = S.copy()
    late.iloc[:20, 2] = np.nan
    constant = late.bfill()

    r_late = algos.SoftBayes(eta=eta).run(late)
    r_constant = algos.SoftBayes(eta=eta).run(constant)

    assert np.isfinite(r_late.B).all().all()
    assert np.allclose(r_late.B.sum(axis=1), 1)
    assert np.allclose(r_late.B, r_constant.B)
    assert np.isclose(r_late.total_wealth, r_constant.total_wealth)


@pytest.mark.parametrize("eta", [None, 0.1])
def test_duplicated_asset(eta, S):
    """Two copies of an asset get equal weights. With a fixed eta the update is linear
    in the weights, so together they behave like one asset with their combined prior
    weight. The anytime eta_t depends on the number of assets, so there they don't."""
    dup = S[["S0", "S0", "S1", "S2"]].set_axis(["A", "B", "S1", "S2"], axis=1)

    class WithPrior(algos.SoftBayes):
        def init_weights(self, columns):
            return np.array([0.5, 0.25, 0.25])

    B_dup = algos.SoftBayes(eta=eta).run(dup).B
    B_one = WithPrior(eta=eta).run(S).B

    assert np.allclose(B_dup["A"], B_dup["B"])
    if eta is None:
        return
    assert np.allclose(B_dup["A"] + B_dup["B"], B_one["S0"])
    assert np.allclose(B_dup[["S1", "S2"]], B_one[["S1", "S2"]])
