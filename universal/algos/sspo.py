import numpy as np

from .. import tools
from ..algo import Algo


class SSPO(Algo):
    """Short-term Sparse Portfolio Optimization. Concentrates wealth on the few assets
    furthest below their recent peak price by solving an l1-regularized linear program
    with ADMM, then projects the scaled solution onto the simplex.

    Reference:
        Zhao-Rong Lai, Pei-Yi Yang, Liangda Fang and Xiaotian Wu.
        Short-term Sparse Portfolio Optimization Based on Alternating Direction
        Method of Multipliers, 2018.
        http://jmlr.org/papers/v19/17-558.html
    """

    PRICE_TYPE = "raw"
    REPLACE_MISSING = True

    def __init__(
        self,
        window=5,
        lam=0.5,
        gamma=0.01,
        eta=0.005,
        zeta=500,
        tol=1e-4,
        max_iter=10000,
        **kwargs,
    ):
        """
        :param window: Lookback window for the peak price.
        :param lam: Strength of the l1 regularization.
        :param gamma: Soft-threshold step, controls how closely the sparse auxiliary
            variable g tracks the weights b.
        :param eta: Penalty strength and dual step for the constraint sum(b) = 1.
        :param zeta: Scale applied to the ADMM solution before the simplex projection.
            Larger values give sparser portfolios.
        :param tol: ADMM stops once |sum(b) - 1| < tol.
        :param max_iter: Maximum number of ADMM iterations per period.
        """
        super().__init__(min_history=1, **kwargs)

        # input check
        if window < 1:
            raise ValueError("window parameter must be >=1")
        if min(lam, gamma, eta, zeta, tol) <= 0:
            raise ValueError("lam, gamma, eta, zeta and tol must be positive")

        self.window = window
        self.lam = lam
        self.gamma = gamma
        self.eta = eta
        self.zeta = zeta
        self.tol = tol
        self.max_iter = max_iter

    def init_weights(self, columns):
        m = len(columns)
        return np.ones(m) / m

    def step(self, x, last_b, history):
        # increasing potential: distance of the current price to its recent peak
        # (eq. 11); with fewer than window + 1 prices use the last price relative,
        # as in the authors' code
        if len(history) < self.window + 1:
            x_pred = history.iloc[-1] / history.iloc[-2]
        else:
            x_pred = history.iloc[-self.window :].max() / history.iloc[-1]
        phi = -(1.1 * np.log(np.asarray(x_pred, dtype=float)) + 1)

        b = self.update(np.asarray(last_b, dtype=float), phi)

        # project it onto simplex (eq. 33)
        return tools.simplex_proj(self.zeta * b)

    def update(self, b, phi):
        """Solve min_b b'phi + lam * ||b||_1 s.t. sum(b) = 1 by ADMM (Algorithm 1),
        warm-started from the last weights."""
        d = len(b)
        tau = self.lam / self.gamma
        eta = self.eta

        # b-step (eq. 29) is (tau*I + eta*11')^-1 (tau*g + (eta - rho)*1 - phi);
        # by Sherman-Morrison it equals g + u + k*1 with u = A^-1 (-phi) fixed for
        # this period and k a scalar depending on sum(g) and rho
        denom = tau * (tau + eta * d)
        u = -phi / tau + eta * phi.sum() / denom

        g = b
        rho = 0.0
        for _ in range(self.max_iter):
            c = eta - rho
            k = c / tau - eta * (tau * g.sum() + d * c) / denom
            b = g + u + k

            # g-step: soft thresholding (eq. 32)
            g = b - np.clip(b, -self.gamma, self.gamma)

            # dual ascent (eq. 27)
            residual = b.sum() - 1
            rho += eta * residual
            if abs(residual) < self.tol:
                break

        return b


if __name__ == "__main__":
    tools.quickrun(SSPO())
