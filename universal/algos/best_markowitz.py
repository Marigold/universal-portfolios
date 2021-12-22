import numpy as np
import pandas as pd

from .. import tools
from .crp import CRP


class BestMarkowitz(CRP):
    """Optimal Markowitz portfolio constructed in hindsight.

    Reference:
        https://en.wikipedia.org/wiki/Modern_portfolio_theory
    """

    PRICE_TYPE = "ratio"
    REPLACE_MISSING = False

    def __init__(self, global_sharpe=None, sharpe=None, **kwargs):
        self.global_sharpe = global_sharpe
        self.sharpe = sharpe
        self.opt_markowitz_kwargs = kwargs

    def weights(self, X):
        """Find optimal markowitz weights."""
        # update frequency
        freq = tools.freq(X.index)

        R = X - 1

        # calculate mean and covariance matrix and annualize them
        sigma = R.cov() * freq

        if self.sharpe:
            mu = pd.Series(np.sqrt(np.diag(sigma)), X.columns) * pd.Series(
                self.sharpe
            ).reindex(X.columns)
        elif self.global_sharpe:
            mu = pd.Series(np.sqrt(np.diag(sigma)) * self.global_sharpe, X.columns)
        else:
            mu = R.mean() * freq

        self.b = tools.opt_markowitz(mu, sigma, **self.opt_markowitz_kwargs)

        return super().weights(R)


if __name__ == "__main__":
    tools.quickrun(BestMarkowitz())
