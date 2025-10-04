import math
from typing import Optional

import numpy as np

from ..algo import Algo


class AdaHedge(Algo):
    """Adaptive Hedge algorithm for online portfolio selection.

    Reference:
        Jyrki Kivinen and Manfred K. Warmuth.
        Averaging Expert Predictions, 1999.
        The adaptive tuning follows de Rooij et al., Follow the Leader If You
        Can, Hedge If You Must (2014).

        https://arxiv.org/abs/1301.0534
    """

    PRICE_TYPE = "ratio"
    REPLACE_MISSING = True

    def __init__(
        self,
        delta0: float = 1e-6,
        min_eta: float = 1e-8,
        max_eta: float = 100.0,
        clip: Optional[float] = None,
    ) -> None:
        """Initialise AdaHedge.

        :param delta0: Initial cumulative mixability gap to avoid division by zero.
        :param min_eta: Lower bound for adaptive learning rate (prevents underflow).
        :param max_eta: Upper bound for adaptive learning rate (prevents overflow).
        :param clip: Optional upper bound for per-asset loss magnitude.
        """
        super().__init__()
        if delta0 <= 0:
            raise ValueError("delta0 must be positive")
        if min_eta <= 0:
            raise ValueError("min_eta must be positive")
        if max_eta <= min_eta:
            raise ValueError("max_eta must exceed min_eta")

        self.delta0 = delta0
        self.min_eta = min_eta
        self.max_eta = max_eta
        self.clip = clip

        self._cum_losses: Optional[np.ndarray] = None
        self._delta: Optional[float] = None

    def init_weights(self, columns):
        m = len(columns)
        return np.ones(m) / m

    def init_step(self, X):
        m = X.shape[1]
        self._cum_losses = np.zeros(m)
        self._delta = self.delta0

    def step(self, x, last_b, history):
        if self._cum_losses is None or self._delta is None:
            raise RuntimeError("AdaHedge state not initialised; call init_step first")

        x_arr = np.asarray(x, dtype=float)
        if np.any(x_arr <= 0):
            raise ValueError("AdaHedge requires strictly positive price relatives")

        losses = -np.log(x_arr)
        if self.clip is not None:
            losses = np.minimum(losses, self.clip)

        m = losses.size
        eta = math.log(m) / max(self._delta, 1e-12)
        eta = float(np.clip(eta, self.min_eta, self.max_eta))

        # unnormalised exponential weights with numerical stabilisation
        logits = -eta * self._cum_losses
        logits -= logits.max()  # shift to improve stability
        unnormalised = np.exp(logits)
        weights = unnormalised / unnormalised.sum()

        # update cumulative losses with current round observation
        self._cum_losses = self._cum_losses + losses

        # compute mixability gap with numerically stable log-sum-exp
        raw_log_terms = -eta * losses
        max_term = float(raw_log_terms.max())
        stable_terms = raw_log_terms - max_term
        exp_terms = np.exp(stable_terms)
        mix_sum = float(np.dot(weights, exp_terms))
        if not math.isfinite(mix_sum) or mix_sum <= 0.0:
            raise RuntimeError("Numerical issue in mixability gap computation")

        h_t = float(np.dot(weights, losses))
        mix_log = max_term + math.log(mix_sum)
        m_t = -mix_log / eta
        delta_increment = max(0.0, h_t - m_t)

        self._delta = max(self.delta0, self._delta + delta_increment)

        return weights


if __name__ == "__main__":
    from .. import tools

    tools.quickrun(AdaHedge())
