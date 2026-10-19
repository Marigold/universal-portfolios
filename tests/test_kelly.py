import numpy as np
import pytest

from universal import algos, tools


@pytest.fixture(scope="module")
def S():
    np.random.seed(0)
    return tools.random_portfolio(n=200, k=3, sd=0.01)


def test_full_kelly(S):
    """Long-short weights are (1 + r) * inv(sigma) @ (mu - r) on the history so far."""
    r = 0.0001
    B = algos.Kelly(r=r).run(S).B
    # B.iloc[-1] is decided after the second-to-last period
    X = algos.Kelly._convert_prices(S, "log").iloc[:-1]
    mu, sigma = X.mean(), X.cov()

    assert np.allclose(B.iloc[-1], (1 + r) * np.linalg.solve(sigma, mu - r))


def test_duplicate_asset(S):
    """An exact duplicate makes sigma singular; the two copies split the weight."""
    S2 = S.assign(dup=S.iloc[:, 0])
    B = algos.Kelly().run(S2).B
    B_ref = algos.Kelly().run(S).B

    assert np.allclose(B.iloc[:, 0], B["dup"])
    assert np.allclose(B.iloc[:, 0] + B["dup"], B_ref.iloc[:, 0])


def test_constant_asset(S):
    """An asset with zero variance gets zero weight."""
    B = algos.Kelly().run(S.assign(const=1.0)).B
    B_ref = algos.Kelly().run(S).B

    assert (B["const"] == 0).all()
    assert np.allclose(B.iloc[:, :3], B_ref)


def test_late_listed_asset(S):
    """An asset without enough history gets zero weight instead of NaN everywhere."""
    S2 = S.copy()
    S2.iloc[:100, 2] = np.nan
    B = algos.Kelly().run(S2).B

    assert B.notnull().all().all()
    assert (B.iloc[:100, 2] == 0).all()


def test_reg_bounds_leverage_of_near_duplicates(S):
    noise = np.exp(np.cumsum(np.random.default_rng(0).normal(0, 1e-6, len(S))))
    S2 = S.assign(near=S.iloc[:, 0] * noise)

    def gross(**kwargs):
        return algos.Kelly(**kwargs).run(S2).B.abs().sum(axis=1).max()

    assert gross() > 1e4
    assert gross(reg=1e-4) < 100
