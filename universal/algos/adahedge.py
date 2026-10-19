import numpy as np

from .. import tools
from ..algo import Algo


class AdaHedge(Algo):
    """AdaHedge: the Hedge algorithm with a self-tuning learning rate, treating every
    asset as an expert whose loss is its negative log return.

    The learning rate eta interpolates between three benchmarks: eta -> 0 is uniform
    CRP, eta = 1 is buy and hold (weights proportional to cumulative return) and
    eta -> inf follows the leader (all-in on the best asset so far). AdaHedge starts
    by following the leader and lowers eta as the cumulative mixability gap grows,
    i.e. as following the leader turns out to be costly. It has no parameters.

    Reference:
        S. de Rooij, T. van Erven, P. D. Grünwald, W. M. Koolen.
        "Follow the Leader If You Can, Hedge If You Must."
        Journal of Machine Learning Research 15 (2014): 1281-1316.
        https://arxiv.org/abs/1301.0534
    """

    PRICE_TYPE = "log"
    REPLACE_MISSING = True

    def init_weights(self, columns):
        m = len(columns)
        return np.ones(m) / m

    def init_step(self, X):
        # cumulative loss of each asset
        self.L = np.zeros(X.shape[1])
        # cumulative mixability gap
        self.Delta = 0.0

    def _eta(self):
        # a single asset has nothing to hedge; rounding could still make Delta > 0,
        # which would give eta = ln(1) / Delta = 0 and a 0 / 0 mix loss
        if self.Delta == 0 or len(self.L) == 1:
            return np.inf
        return np.log(len(self.L)) / self.Delta

    def _mix(self, eta, L):
        """Return weights and mix loss M for cumulative losses L."""
        mn = L.min()
        w = (L == mn).astype(float) if eta == np.inf else np.exp(-eta * (L - mn))
        s = w.sum()
        return w / s, mn - np.log(s / len(L)) / eta

    def step(self, x, last_b, history=None):
        loss = -np.asarray(x, dtype=float)

        # weights held during this period, computed by the previous step
        eta = self._eta()
        w, M_prev = self._mix(eta, self.L)
        self.L = self.L + loss
        _, M = self._mix(eta, self.L)

        # the gap is nonnegative by Jensen's inequality; max only absorbs rounding
        self.Delta += max(0.0, w @ loss - (M - M_prev))

        b, _ = self._mix(self._eta(), self.L)
        return b


if __name__ == "__main__":
    tools.quickrun(AdaHedge())
