import numpy as np

from .. import tools
from ..algo import Algo


class SoftBayes(Algo):
    """Soft-Bayes: each period, move the weights a fraction eta of the way from where
    they are to where last period's returns let them drift,

        b_{t+1} = b_t * (1 - eta + eta * x_t / (b_t @ x_t)).

    eta = 1 is buy and hold (the Bayesian posterior over assets, hence the name) and
    eta = 0 is uniform CRP. It uses the same signal as EG, but the update is linear
    instead of exponential, so one period can add at most eta to an asset's weight and
    remove at most a fraction eta of it. That makes its regret against the best CRP
    independent of how extreme the price relatives get, which is what breaks EG. It
    costs O(N) per step for N assets.

    The default, eta=None, is the paper's anytime variant: a learning rate that
    decreases as sqrt(ln N / (2 N t)), with a small pull back towards the initial
    weights, so it needs neither tuning nor the length of the history.

    What it is good for: a cheap, robust baseline that is guaranteed to stay close to
    the best CRP. With the learning rates the theory recommends it hardly moves away
    from uniform weights, so on real data it is hard to tell from uniform CRP.

    The math, its guarantees and how it maps to this code are in soft_bayes.md next to
    this file.

    Reference:
        L. Orseau, T. Lattimore, S. Legg.
        "Soft-Bayes: Prod for Mixtures of Experts with Log-Loss."
        Algorithmic Learning Theory (ALT), 2017.
        https://arxiv.org/abs/1901.02230
    """

    REPLACE_MISSING = True

    def __init__(self, eta=None):
        """
        :param eta: Fixed learning rate in [0, 1], or None for the anytime learning
            rate eta_t = sqrt(ln(N) / (2 * N * t)) of Section 6. With N assets and T
            periods known in advance, eta / (1 - eta) = sqrt(ln(N) / (T * N)) bounds the
            log-wealth regret against the best CRP by 2 * sqrt(T * N * ln(N)) + ln(N).
        """
        super().__init__()
        if eta is not None and not 0 <= eta <= 1:
            raise ValueError(f"eta must be in [0, 1] or None, got {eta}")
        self.eta = eta

    def init_weights(self, columns):
        m = len(columns)
        return np.ones(m) / m

    def init_step(self, X):
        # prior weights w_1, which the anytime variant mixes back in
        self.w1 = np.asarray(self.init_weights(X.columns), dtype=float)
        # number of rounds seen so far, for the anytime learning rate
        self.t = 0

    def step(self, x, last_b, history=None):
        b = np.asarray(last_b, dtype=float)
        x = np.asarray(x, dtype=float)
        self.t += 1
        n_assets = len(b)

        if self.eta is None:
            eta = np.sqrt(np.log(n_assets) / (2 * n_assets * self.t))
            # online correction of eq. (14); eta_{t+1} / eta_t = sqrt(t / (t + 1)),
            # written out so that it is defined for a single asset (eta_t = 0)
            beta = np.sqrt(self.t / (self.t + 1))
        else:
            eta, beta = self.eta, 1.0

        # eq. (7), then the correction (a no-op for a fixed eta)
        b = b * (1 - eta + eta * x / (b @ x))
        return beta * b + (1 - beta) * self.w1


if __name__ == "__main__":
    tools.quickrun(SoftBayes())
