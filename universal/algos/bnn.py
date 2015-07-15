# -*- coding: utf-8 -*-
from ..algo import Algo
import numpy as np
import pandas as pd
from .. import tools


class BNN(Algo):
    """ Nearest neighbor based strategy. It tries to find similar sequences of price in history and
    then maximize objective function (that is profit) on the days following them.

    Reference:
        L. Gyorfi, G. Lugosi, and F. Udina. Nonparametric kernel based sequential
        investment strategies. Mathematical Finance 16 (2006) 337–357.
    """

    PRICE_TYPE = 'ratio'
    REPLACE_MISSING = True

    def __init__(self, k=5, l=10):
        """
        :param k: Sequence length.
        :param l: Number of nearest neighbors.
        """

        super(BNN, self).__init__(min_history=k+l-1)

        self.k = k
        self.l = l


    def init_weights(self, m):
        return np.ones(m) / m


    def step(self, x, last_b, history):
        # find indices of nearest neighbors throughout history
        ixs = self.find_nn(history, self.k, self.l)

        # get returns from the days following NNs
        J = history.iloc[[history.index.get_loc(i) + 1 for i in ixs]]

        # get best weights
        return tools.bcrp_weights(J)


    def find_nn(self, H, k, l):
        """ Note that nearest neighbors are calculated in a different (more efficient) way than shown
        in the article.

        param H: history
        """
        # calculate distance from current sequence to every other point
        D = H * 0
        for i in range(1, k+1):
            D += (H.shift(i-1) - H.iloc[-i])**2
        D = D.sum(1).iloc[:-1]

        # sort and find nearest neighbors
        D.sort()
        return D.index[:l]


if __name__ == '__main__':
    tools.quickrun(BNN())
