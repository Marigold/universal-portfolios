import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import tools
from ..algo import Algo


class CRP(Algo):
    """Constant rebalanced portfolio = use fixed weights all the time. Uniform weights
    are commonly used as a benchmark.

    Reference:
        T. Cover. Universal Portfolios, 1991.
        http://www-isl.stanford.edu/~cover/papers/paper93.pdf
    """

    def __init__(self, b=None):
        """
        :params b: Constant rebalanced portfolio weights. Default is uniform.
        """
        super().__init__()
        self.b = np.array(b) if b is not None else None

    def step(self, x, last_b, history):
        # init b to default if necessary
        if self.b is None:
            self.b = np.ones(len(x)) / len(x)
        return self.b

    def weights(self, X):
        if self.b is None:
            b = X * 0 + 1
            b.loc[:, "CASH"] = 0
            b = b.div(b.sum(axis=1), axis=0)
            return b
        elif self.b.ndim == 1:
            return np.repeat([self.b], X.shape[0], axis=0)
        else:
            return self.b

    @classmethod
    def plot_crps(cls, data, show_3d=False):
        """Plot performance graph for all CRPs (Constant Rebalanced Portfolios).
        :param data: Stock prices.
        :param show_3d: Show CRPs on a 3-simplex, works only for 3 assets.
        """

        def _crp(data):
            B = list(tools.simplex_mesh(2, 100))
            crps = CRP.run_combination(data, b=B)
            x = [b[0] for b in B]
            y = [c.total_wealth for c in crps]
            return x, y

        # init
        import ternary

        data = data.dropna(how="any")
        data = data / data.iloc[0]
        dim = data.shape[1]

        # plot prices
        if dim == 2 and not show_3d:
            fig, axes = plt.subplots(ncols=2, sharey=True)
            data.plot(ax=axes[0], logy=True)
        else:
            data.plot(logy=False)

        if show_3d:
            assert dim == 3, "3D plot works for exactly 3 assets."
            plt.figure()
            fun = lambda b: CRP(b).run(data).total_wealth
            ternary.plot_heatmap(fun, steps=20, boundary=True)

        elif dim == 2:
            x, y = _crp(data)
            s = pd.Series(y, index=x)
            s.plot(ax=axes[1], logy=True)
            plt.title("CRP performance")
            plt.xlabel("weight of {}".format(data.columns[0]))

        elif dim > 2:
            fig, axes = plt.subplots(ncols=dim - 1, nrows=dim - 1)
            for i in range(dim - 1):
                for j in range(i + 1, dim):
                    x, y = _crp(data[[i, j]])
                    ax = axes[i][j - 1]
                    ax.plot(x, y)
                    ax.set_title("{} & {}".format(data.columns[i], data.columns[j]))
                    ax.set_xlabel("weights of {}".format(data.columns[i]))


if __name__ == "__main__":
    result = tools.quickrun(CRP())
    print(result.information)
