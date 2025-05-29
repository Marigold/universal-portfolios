import numpy as np
import pandas as pd
from cvxopt import solvers

from .. import tools
from ..algo import Algo

solvers.options["show_progress"] = False


class Kelly(Algo):
    """Kelly fractioned betting. See
    http://en.wikipedia.org/wiki/Kelly_criterion#Application_to_the_stock_market
    for quick introduction.
    """

    PRICE_TYPE = "log"
    REPLACE_MISSING = False

    def __init__(
        self,
        window=float("inf"),
        r=0.0,
        fraction=1.0,
        long_only=False,
        min_history=None,
        max_leverage=1.0,
        reg=0.0,
        q=1.0,
        mu_estimate=False,
        gamma=0.0,
    ):
        """
        :param window: Window for calculating mean and variance. Use float('inf') for entire history.
        :param min_history: Use zero weights for first min_periods.
        :param r: Risk-free rate.
        :param long_only: Restrict to positive portfolio weights.
        :param fraction: Use fraction of Kelly weights. 1. is full Kelly, 0.5 is half Kelly.
        :param max_leverage: Max leverage to use.
        :param reg: Regularization parameter for covariance matrix (adds identity matrix).
        :param mu_estimate: Mean is estimated to be proportional to historical variance
        :param gamma: Penalize changing weights.
        """
        if np.isinf(window):
            window = int(1e8)
            min_history = min_history or 50
        else:
            min_history = min_history or window

        super().__init__(min_history=min_history)
        self.window = window
        self.r = r
        self.fraction = fraction
        self.long_only = long_only
        self.max_leverage = max_leverage
        self.reg = reg
        self.q = q
        self.mu_estimate = mu_estimate
        self.gamma = gamma

    def init_step(self, X):
        # precalculate correlations
        self.S = tools.rolling_cov_pairwise(
            X, window=self.window, min_periods=self.min_history
        )
        self.M = X.rolling(window=self.window, min_periods=self.min_history).mean()

    def step(self, x, last_b, history):
        # get sigma and mu matrix
        mu = self.M.loc[x.name]
        sigma = self.S.loc[x.name]

        # make sure sigma is properly indexed
        sigma = sigma.reindex(index=x.index).reindex(columns=x.index)

        # mu is proportional to individual variance
        if self.mu_estimate:
            mu = pd.Series(np.sqrt(np.diag(sigma)), index=mu.index)

        # penalize changing weights
        m = len(mu)
        gamma = self.gamma
        q = self.q
        if gamma != 0:
            sigma += gamma * np.eye(m)
            if q == 0:
                mu = 2.0 * gamma * last_b
            else:
                mu += 2.0 * gamma / q

        # pure approach - problems with singular matrix
        if not self.long_only:
            sigma = np.matrix(sigma)
            mu = np.matrix(mu).T

            sigma_inv = np.linalg.inv(sigma)
            b = (1 + self.r) * sigma_inv * (mu - self.r)
            b = np.ravel(b)
        else:
            b = tools.opt_markowitz(
                mu,
                sigma,
                long_only=self.long_only,
                reg=self.reg,
                rf_rate=self.r,
                q=self.q,
                max_leverage=self.max_leverage,
            )

        # use Kelly fraction
        b *= self.fraction

        return b

    def plot_fraction(self, S, fractions=np.linspace(0.0, 2.0, 10), **kwargs):
        """Plot graph with Kelly fraction on x-axis and total wealth on y-axis.
        :param S: Stock prices.
        :param fractions: List (ndarray) of fractions used.
        """
        wealths = []
        for fraction in fractions:
            self.fraction = fraction
            wealths.append(self.run(S).total_wealth)

        ax = pd.Series(wealths, index=fractions, **kwargs).plot(**kwargs)
        ax.set_xlabel("Kelly Fraction")
        ax.set_ylabel("Total Wealth")
        return ax


# use case
if __name__ == "__main__":
    tools.quickrun(Kelly())
