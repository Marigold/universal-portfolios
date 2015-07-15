from ..algo import Algo
import numpy as np
import pandas as pd
from .. import tools


class PAMR(Algo):
    """ Passive aggressive mean reversion strategy for portfolio selection.
    There are three variants with different parameters, see original article
    for details.

    Reference:
        B. Li, P. Zhao, S. C.H. Hoi, and V. Gopalkrishnan.
        Pamr: Passive aggressive mean reversion strategy for portfolio selection, 2012.
        http://www.cais.ntu.edu.sg/~chhoi/paper_pdf/PAMR_ML_final.pdf
    """

    PRICE_TYPE = 'ratio'
    REPLACE_MISSING = True

    def __init__(self, eps=0.5, C=500, variant=0):
        """
        :param eps: Control parameter for variant 0. Must be >=0, recommended value is
                    between 0.5 and 1.
        :param C: Control parameter for variant 1 and 2. Recommended value is 500.
        :param variant: Variants 0, 1, 2 are available.
        """
        super(PAMR, self).__init__()

        # input check
        if not(eps >= 0):
            raise ValueError('epsilon parameter must be >=0')

        if variant == 0:
            if eps is None:
                raise ValueError('eps parameter is required for variant 0')
        elif variant == 1 or variant == 2:
            if C is None:
                raise ValueError('C parameter is required for variant 1,2')
        else:
            raise ValueError('variant is a number from 0,1,2')

        self.eps = eps
        self.C = C
        self.variant = variant


    def init_weights(self, m):
        return np.ones(m) / m


    def step(self, x, last_b):
        # calculate return prediction
        b = self.update(last_b, x, self.eps, self.C)
        return b


    def update(self, b, x, eps, C):
        """ Update portfolio weights to satisfy constraint b * x <= eps
        and minimize distance to previous weights. """
        x_mean = np.mean(x)
        le = max(0., np.dot(b, x) - eps)

        if self.variant == 0:
            lam = le / np.linalg.norm(x - x_mean)**2
        elif self.variant == 1:
            lam = min(C, le / np.linalg.norm(x - x_mean)**2)
        elif self.variant == 2:
            lam = le / (np.linalg.norm(x - x_mean)**2 + 0.5 / C)

        # limit lambda to avoid numerical problems
        lam = min(100000, lam)

        # update portfolio
        b = b - lam * (x - x_mean)

        # project it onto simplex
        return tools.simplex_proj(b)


if __name__ == '__main__':
    tools.quickrun(PAMR())





