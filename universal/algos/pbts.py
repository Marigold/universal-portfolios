import numpy as np
from cvxopt import matrix, solvers
from sklearn import covariance
from sklearn.base import BaseEstimator

from .. import tools
from ..algo import Algo

BENCHMARKS = ("EW", "VW")

# Returns of the two blends closer than this count as equal (footnote 1). Weights that
# sum to one only up to rounding, as on the first period's "no change" returns, would
# otherwise decide a test by rounding error.
TIE_TOLERANCE = 1e-12

COV_ESTIMATORS = {
    "empirical": covariance.EmpiricalCovariance,
    "ledoit-wolf": covariance.LedoitWolf,
    "oas": covariance.OAS,
}


class PBTS(Algo):
    """Portfolio Blending via Thompson Sampling: hold a blend of the minimum-variance
    portfolio and a benchmark portfolio, delta * MV + (1 - delta) * benchmark, where the
    blending coefficient delta is learned each period from the last tau periods.

    With benchmark="EW" (equally weighted) this is the paper's TS-EM portfolio, and with
    benchmark="VW" (value weighted) it is TS-VM. delta gets a Beta(1, 1) prior that is
    restarted every period. Each of the last tau periods is a Bernoulli test: a sampled
    delta is compared with the prior mean, and the test counts as a success when moving
    towards MV in the sampled direction would have paid off. delta is the posterior mean
    after the window.

    One consequence the paper doesn't state: the sampled delta cancels out of the success
    test, so a period counts as a success exactly when MV beat the benchmark in it, and
    delta = (1 + wins) / (2 + decided periods). The algorithm is deterministic given the
    data, and the random seed only matters for ties. pbts.md works this out.

    Departures from the paper, all documented in pbts.md:
    - The minimum-variance portfolio is long-only (the paper's closed form, eq. 4, can
      short), as preferred by the library's maintainer.
    - The covariance is the empirical one by default (the paper uses a factor model).
    - Without market capitalisations, VW is buy and hold from uniform weights.
    - Until tau periods have been seen, it holds the benchmark portfolio. The paper
      assumes tau periods of training data before the first decision.

    What it is good for: a cautious blend of a low-risk portfolio and a naive one, aimed
    at risk-adjusted returns rather than wealth. It doesn't try to follow winners or
    losers, so it is closer to MPT(method="variance") and uniform CRP than to most
    algorithms in this package. The math, the paper's claims and how they map to this
    code are in pbts.md next to this file.

    Reference:
        W. Shen, J. Wang.
        "Portfolio Blending via Thompson Sampling."
        International Joint Conference on Artificial Intelligence (IJCAI), 2016.
        https://www.ijcai.org/Proceedings/16/Papers/283.pdf
    """

    PRICE_TYPE = "ratio"
    REPLACE_MISSING = True

    def __init__(self, benchmark="EW", tau=120, cov_estimator="empirical", seed=0):
        """
        :param benchmark: "EW" for the equally weighted portfolio (TS-EM) or "VW" for the
            value-weighted one (TS-VM), which without market capitalisations is buy and
            hold from uniform weights.
        :param tau: Number of past periods used to estimate the covariance and to run
            the Bernoulli tests. The paper uses 120 months or 200 weeks.
        :param cov_estimator: "empirical", "ledoit-wolf", "oas", or any scikit-learn
            covariance estimator. It is fitted on the net returns of the last tau periods.
        :param seed: Seed for the Beta samples. It only affects how exact ties are broken.
        """
        super().__init__()
        if benchmark not in BENCHMARKS:
            raise ValueError(
                f"benchmark must be one of {BENCHMARKS}, got {benchmark!r}"
            )
        if int(tau) != tau or tau < 2:
            raise ValueError(f"tau must be an integer of at least 2, got {tau!r}")
        if isinstance(cov_estimator, str):
            if cov_estimator not in COV_ESTIMATORS:
                raise ValueError(
                    f"cov_estimator must be one of {list(COV_ESTIMATORS)} or a "
                    f"scikit-learn estimator, got {cov_estimator!r}"
                )
            cov_estimator = COV_ESTIMATORS[cov_estimator]()
        elif not isinstance(cov_estimator, BaseEstimator):
            raise ValueError("cov_estimator must be a name or a scikit-learn estimator")
        self.benchmark = benchmark
        self.tau = int(tau)
        self.cov_estimator = cov_estimator
        self.seed = seed

    def init_weights(self, columns):
        m = len(columns)
        return np.ones(m) / m

    def init_step(self, X):
        n_assets = X.shape[1]
        # gross returns R_k of the periods seen so far, oldest first, at most tau of them
        self.returns = []
        # value-weighted portfolio, eq. (3); starts from uniform weights
        self.vw = np.ones(n_assets) / n_assets
        self.rng = np.random.default_rng(self.seed)

    def step(self, x, last_b, history=None):
        x = np.asarray(x, dtype=float)
        n_assets = len(x)

        # eq. (3): the value-weighted portfolio drifts with prices
        self.vw = self.vw * x / (self.vw @ x)

        self.returns.append(x)
        if len(self.returns) > self.tau:
            self.returns.pop(0)

        # eq. (2) or (3)
        benchmark = (
            np.ones(n_assets) / n_assets if self.benchmark == "EW" else self.vw.copy()
        )
        if len(self.returns) < self.tau or n_assets == 1:
            return benchmark

        R = np.array(self.returns)
        mv = self.min_variance(R)
        delta = self.blending_coefficient(R, mv, benchmark)
        # eq. (11)
        return delta * mv + (1 - delta) * benchmark

    def min_variance(self, R):
        """Long-only minimum-variance portfolio for the covariance of the window (eq. 4
        with the constraint b >= 0 added)."""
        sigma = self.cov_estimator.fit(R - 1).covariance_
        return long_only_min_variance(sigma)

    def blending_coefficient(self, R, mv, benchmark):
        """Algorithm 1, lines 6 to 18: Bernoulli tests over the window R, oldest first."""
        a, b = 1, 1
        for r in R:
            # eq. (7): benchmark blend at the current Beta mean
            delta_bar = a / (a + b)
            omega_bar = delta_bar * mv + (1 - delta_bar) * benchmark
            # eq. (8): test blend at a sampled delta; resample on a tie (footnote 1)
            delta_tilde = self.rng.beta(a, b)
            while delta_tilde == delta_bar:
                delta_tilde = self.rng.beta(a, b)
            omega_tilde = delta_tilde * mv + (1 - delta_tilde) * benchmark
            # R'omega_tilde - R'omega_bar, computed directly so that rounding doesn't
            # turn a tie into a win or a loss
            gain = r @ (omega_tilde - omega_bar)
            # footnote 1: no update when the two blends earned the same
            if abs(gain) <= TIE_TOLERANCE:
                continue
            # eq. (9)
            if (gain > 0) == (delta_tilde > delta_bar):
                a += 1
            else:
                b += 1
        # eq. (10): posterior mean of Beta(1 + successes, 1 + failures)
        return a / (a + b)


def long_only_min_variance(sigma):
    """Minimize b' sigma b subject to b >= 0 and sum(b) = 1."""
    n = sigma.shape[0]
    P = matrix(2 * np.asarray(sigma, dtype=float))
    q = matrix(np.zeros(n))
    G = matrix(-np.eye(n))
    h = matrix(np.zeros(n))
    A = matrix(np.ones((1, n)))
    b = matrix(np.ones(1))
    sol = solvers.qp(P, q, G, h, A, b, options={"show_progress": False})
    w = np.clip(np.asarray(sol["x"], dtype=float).ravel(), 0, None)
    return w / w.sum()


if __name__ == "__main__":
    tools.quickrun(PBTS(tau=60))
