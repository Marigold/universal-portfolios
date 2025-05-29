import numpy as np
import pandas as pd

from .. import tools
from ..algo import Algo


class UP(Algo):
    """Universal Portfolio by Thomas Cover.

    Reference:
        T. Cover. Universal Portfolios, 1991.
        http://www-isl.stanford.edu/~cover/papers/paper93.pdf
    """

    PRICE_TYPE = "ratio"
    REPLACE_MISSING = True

    def __init__(self, eval_points=1e4):
        """
        :param eval_points: Number of evaluated points (approximately). Complexity of the
            algorithm is O(eval_points * dim**2).
        """
        super(UP, self).__init__()
        self.eval_points = eval_points

    def init_weights(self, columns):
        """Create a mesh on simplex and keep wealth of all strategies."""
        # create set of CRPs
        self.W = np.matrix(tools.simplex_mesh(len(columns), self.eval_points))
        self.S = np.matrix(np.ones(self.W.shape[0])).T

        # calculate integral with trapezoid rule - weight of a point is
        # a number of neighbors here
        self.TRAP = np.sum(self.W != 0, axis=1)

        # start with uniform weights
        return np.ones(len(columns)) / len(columns)

    def step(self, x, last_b, history=None):
        # calculate new wealth of all CRPs
        self.S = np.multiply(self.S, self.W * np.matrix(x).T)
        b = self.W.T * np.multiply(self.S, self.TRAP)
        return b / sum(b)


# use case
if __name__ == "__main__":
    import random

    import matplotlib.pyplot as plt

    random.seed(42)
    data = pd.read_pickle("../data/nyse_o.pd")
    data = data[random.sample(data.columns, 3)]

    result = UP(eval_points=1e3).run(data)
    print(result.summary())
    ax1 = result.plot(assets=True, weights=True, logy=False)
    ax1.set_title("Nice one")
    plt.show()
