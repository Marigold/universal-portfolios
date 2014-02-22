# -*- coding: utf-8 -*-
from universal.algo import Algo
import universal.tools as tools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class EG(Algo):
    """ Exponentiated Gradient (EG) algorithm by Helmbold et al. 

    Reference:
        Helmbold, David P., et al. 
        "On‐Line Portfolio Selection Using Multiplicative Updates." 
        Mathematical Finance 8.4 (1998): 325-347.
    """

    def __init__(self, eta=0.05):
        """
        :params eta: Learning rate. Controls volatility of weights.
        """
        super(EG, self).__init__()
        self.eta = eta


    def init_weights(self, m):
        return np.ones(m) / m


    def step(self, x, last_b):
        b = last_b * np.exp(self.eta * x / sum(x * last_b))
        return b / sum(b)


    @classmethod
    def plot_crps(cls, data, show_3d=False):
        """ Plot performance graph for all CRPs (Constant Rebalanced Portfolios).
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
        data = data.dropna(how='any')
        data = data / data.ix[0]
        dim = data.shape[1]
        
        # plot prices
        if dim == 2 and not show_3d:
            fig, axes = plt.subplots(ncols=2, sharey=True)
            data.plot(ax=axes[0], logy=False)
        else:
            data.plot(logy=False)
        
        if show_3d:
            assert dim == 3, '3D plot works for exactly 3 assets.'
            plt.figure()
            fun = lambda b: CRP(b).run(data).total_wealth
            ternary.plot_heatmap(fun, steps=20, boundary=True)
            
        elif dim == 2:
            x,y = _crp(data)
            s = pd.Series(y, index=x)
            s.plot(ax=axes[1])
            plt.title('CRP performance')
            plt.xlabel('weight of {}'.format(data.columns[0]))
            
        elif dim > 2:
            fig, axes = plt.subplots(ncols=dim-1, nrows=dim-1)
            for i in range(dim-1):
                for j in range(i + 1, dim):
                    x,y = _crp(data[[i,j]])
                    ax = axes[i][j-1] 
                    ax.plot(x,y)
                    ax.set_title('{} & {}'.format(data.columns[i], data.columns[j]))
                    ax.set_xlabel('weights of {}'.format(data.columns[i]))


if __name__ == '__main__':
    tools.quickrun(EG())