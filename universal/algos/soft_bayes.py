import numpy as np

from .. import tools
from ..algo import Algo


class SoftBayes(Algo):
    """Soft-Bayes algorithm by Orseau, Lattimore and Legg. Moves weights by a fraction eta
    of the drift caused by last period's returns, so eta=1 is BAH and eta=0 is uniform
    CRP. Unlike EG, the regret against BCRP doesn't depend on the largest price relative.

    Reference:
        L. Orseau, T. Lattimore, S. Legg.
        Soft-Bayes: Prod for Mixtures of Experts with Log-Loss, 2017.
        https://arxiv.org/abs/1901.02230
    """

    def __init__(self, eta=0.01):
        """
        :param eta: Learning rate in [0, 1]. With m assets and n periods, eta / (1 - eta) =
            sqrt(ln(m) / (n * m)) bounds log-regret against BCRP by
            2 * sqrt(n * m * ln(m)) + ln(m).
        """
        super().__init__()
        self.eta = eta

    def init_weights(self, columns):
        m = len(columns)
        return np.ones(m) / m

    def step(self, x, last_b, history):
        return last_b * (1 - self.eta + self.eta * x / np.dot(last_b, x))


if __name__ == "__main__":
    data = tools.dataset("nyse_o")
    tools.quickrun(SoftBayes(), data)
