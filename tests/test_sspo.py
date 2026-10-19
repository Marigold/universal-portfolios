import numpy as np
import pandas as pd
import pytest

from universal import algos, tools


def _admm_dense(algo, b0, phi):
    """Algorithm 1, steps 3-7, with the b-step (eq. 29) solved with the full matrix."""
    d = len(b0)
    tau = algo.lam / algo.gamma
    A = tau * np.eye(d) + algo.eta * np.ones((d, d))
    b = g = b0
    rho = 0.0
    for _ in range(algo.max_iter):
        b = np.linalg.solve(A, tau * g + (algo.eta - rho) - phi)
        g = np.sign(b) * np.maximum(np.abs(b) - algo.gamma, 0)
        rho += algo.eta * (b.sum() - 1)
        if abs(b.sum() - 1) < algo.tol:
            break
    return b


def _authors_run(data, algo):
    """Port of the authors' SSPO_run.m and SSPO_fun.m (github.com/laizhr/SSPO) on price
    relatives `data`. Returns the weights held on each day."""
    w = algo.window
    T, d = data.shape
    close = np.cumprod(np.vstack([np.ones(d), data[1:]]), axis=0)
    B = np.zeros((T, d))
    port = np.ones(d) / d
    for t in range(1, T + 1):  # MATLAB's 1-based day
        B[t - 1] = port
        if t < T:
            # SSPO_fun with tplus1 = t + 1
            if t + 1 < w + 2:
                x = data[t - 1]
            else:
                x = close[t - w : t].max(axis=0) / close[t - 1]
            phi = -(1.1 * np.log(x) + 1)
            port = tools.simplex_proj(algo.zeta * _admm_dense(algo, port, phi))
    return B


def test_admm_matches_closed_form():
    """The Sherman-Morrison b-step gives the same ADMM solution as eq. (29) solved with
    the full matrix."""
    algo = algos.SSPO()
    rng = np.random.default_rng(0)
    d = 7
    phi = -(1.1 * np.log(1 + rng.uniform(0, 0.1, d)) + 1)
    b0 = np.ones(d) / d

    assert np.allclose(algo.update(b0, phi), _admm_dense(algo, b0, phi))


def test_first_admm_iteration_by_hand():
    """From b = g = 1/d and rho = 0 with all R equal, eq. (29) gives
    (tau/d + eta + R) / (tau + eta * d) for every asset, with tau = lambda / gamma."""
    algo = algos.SSPO(max_iter=1)
    d, R = 4, 1.2
    tau = algo.lam / algo.gamma
    expected = (tau / d + algo.eta + R) / (tau + algo.eta * d)

    assert np.allclose(algo.update(np.ones(d) / d, -R * np.ones(d)), expected)


def test_holds_asset_with_highest_increasing_potential():
    """Weights held in period t + 1 go all-in on the asset with the highest R_t (eq. 11)
    computed from prices up to t: the last price relative while there are at most w prices
    (the authors' warm-up), then the distance to the peak of the last w prices."""
    np.random.seed(0)
    S = tools.random_portfolio(n=60, k=5, sd=0.03)
    P = S.to_numpy()
    w = 5
    B = algos.SSPO(window=w).run(S).B.to_numpy()

    def R(t, warmup):
        x = P[t] / P[t - 1] if warmup else P[t - w + 1 : t + 1].max(axis=0) / P[t]
        return 1.1 * np.log(x) + 1

    # on this data the two rules disagree on the first day of the peak rule
    assert np.argmax(R(w, True)) != np.argmax(R(w, False))

    for t in range(1, len(P) - 1):
        assert B[t + 1].max() > 0.99
        assert np.argmax(B[t + 1]) == np.argmax(R(t, t < w))


def test_matches_reference_code():
    """Same weights as the authors' MATLAB code fed with the price relatives the library
    sees; the first row of S is the base price, so the first relative is 1."""
    np.random.seed(1)
    S = tools.random_portfolio(n=80, k=5, sd=0.02)
    algo = algos.SSPO()
    data = (S / S.shift(1)).fillna(1.0).to_numpy()

    assert np.allclose(algo.run(S).B, _authors_run(data, algo), atol=1e-8)


def test_small_zeta_is_uniform_crp(S):
    """Projecting zeta * b onto the simplex (eq. 33) gives uniform weights as zeta -> 0."""
    B = algos.SSPO(zeta=1e-9).run(S).B
    assert np.allclose(B, 1 / S.shape[1])


def test_sparsity_matches_paper():
    """Average sparsity (eq. 34) of the ADMM output before the projection on DJIA is
    91.91% in Table 1 of the paper."""

    class Sparsity(algos.SSPO):
        def init_step(self, X):
            self.spar = []

        def update(self, b, phi):
            b = super().update(b, phi)
            i = np.argmax(b)
            self.spar.append(np.mean(np.delete(b, i) <= 0.1 * b[i]))
            return b

    algo = Sparsity()
    algo.run(tools.dataset("djia"))

    assert np.mean(algo.spar) == pytest.approx(0.9191, abs=0.005)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_single_asset():
    S = tools.random_portfolio(n=30, k=1)
    assert (algos.SSPO().run(S).B == 1).all().all()


def test_constant_prices():
    """With all prices constant every asset has R = 1, so weights stay uniform."""
    S = pd.DataFrame(np.ones((20, 3)))
    assert np.allclose(algos.SSPO().run(S).B, 1 / 3)


def test_constant_asset_is_not_held():
    """A constant price is always at its peak, R = 1 is the lowest possible score under
    the peak rule, so the asset is never held once that rule applies."""
    np.random.seed(0)
    S = tools.random_portfolio(n=100, k=3, sd=0.02)
    S["const"] = 1.0
    B = algos.SSPO().run(S).B

    assert (B["const"].iloc[6:] == 0).all()


def test_late_listed_asset():
    """Before listing, missing prices are filled with "no change", so the asset looks
    constant and is not held under the peak rule; after listing it is traded."""
    np.random.seed(0)
    S = tools.random_portfolio(n=200, k=3, sd=0.02)
    S.iloc[:100, 0] = np.nan
    B = algos.SSPO().run(S).B

    assert np.isfinite(B).all().all()
    assert (B.iloc[6:100, 0] == 0).all()
    assert (B.iloc[100:, 0] > 0.5).any()


def test_duplicated_asset():
    """Two copies of an asset get the same weight."""
    np.random.seed(0)
    S = tools.random_portfolio(n=100, k=3, sd=0.02)
    S["dup"] = S.iloc[:, 0]
    B = algos.SSPO().run(S).B

    assert np.allclose(B.iloc[:, 0], B["dup"])
    assert (B["dup"] > 0.4).any()
