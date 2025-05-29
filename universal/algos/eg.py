# -*- coding: utf-8 -*-
import numpy as np

from .. import tools
from ..algo import Algo


class EG(Algo):
    """Exponentiated Gradient (EG) algorithm by Helmbold et al.

    Reference:
        Helmbold, David P., et al.
        "On‐Line Portfolio Selection Using Multiplicative Updates."
        Mathematical Finance 8.4 (1998): 325-347.
    """

    def __init__(self, eta=0.05):
        """
        :params eta: Learning rate. Controls volatility of weights.
        """
        super().__init__()
        self.eta = eta

    def init_weights(self, columns):
        m = len(columns)
        return np.ones(m) / m

    def step(self, x, last_b, history):
        b = last_b * np.exp(self.eta * x / sum(x * last_b))
        return b / sum(b)


if __name__ == "__main__":
    data = tools.dataset("nyse_o")
    tools.quickrun(EG(eta=0.5), data)
