# -*- coding: utf-8 -*-
from universal.algo import Algo
from universal.algos import CRP
import universal.tools as tools
import numpy as np


class BCRP(CRP):
    """ Best Constant Rebalanced Portfolio = Constant Rebalanced Portfolio constructed
    with hindsight. It is often used as benchmark.

    Reference:
        T. Cover. Universal Portfolios, 1991.
        http://www-isl.stanford.edu/~cover/papers/paper93.pdf
    """

    def weights(self, X):
        """ Find weights which maximize return on X in hindsight! """
        self.b = tools.bcrp_weights(X)
        return super(BCRP, self).weights(X)


if __name__ == '__main__':
    tools.quickrun(BCRP())
    