from ..algo import Algo
from .. import tools
import numpy as np
import pandas as pd
from numpy import matrix
from cvxopt import solvers, matrix
solvers.options['show_progress'] = False


class CORN(Algo):
    """
    Correlation-driven nonparametric learning approach. Similar to anticor but instead
    of distance of return vectors they use correlation.
    In appendix of the article, universal property is proven.

    Two versions are available. Fast which provides around 2x speedup, but uses more memory
    (linear in window) and slow version which is memory efficient. Most efficient would
    be to rewrite it in sweave or numba.

    Reference:
        B. Li, S. C. H. Hoi, and V. Gopalkrishnan.
        Corn: correlation-driven nonparametric learning approach for portfolio selection, 2011.
        http://www.cais.ntu.edu.sg/~chhoi/paper_pdf/TIST-CORN.pdf
    """

    PRICE_TYPE = 'ratio'
    REPLACE_MISSING = True

    def __init__(self, window=5, rho=0.1, fast_version=True):
        """
        :param window: Window parameter.
        :param rho: Correlation coefficient threshold. Recommended is 0.
        :param fast_version: If true, use fast version which provides around 2x speedup, but uses
                             more memory.
        """
        # input check
        if not(-1 <= rho <= 1):
            raise ValueError('rho must be between -1 and 1')
        if not(window >= 2):
            raise ValueError('window must be greater than 2')

        super(CORN, self).__init__()
        self.window = window
        self.rho = rho
        self.fast_version = fast_version

        # assign step method dynamically
        self.step = self.step_fast if self.fast_version else self.step_slow


    def init_weights(self, m):
        return np.ones(m) / m


    def init_step(self, X):
        if self.fast_version:
            # redefine index to enumerate
            X.index = range(len(X))

            foo = [X.shift(i) for i in range(self.window)]
            self.X_flat = pd.concat(foo, axis=1)
            self.X = X
            self.t = -1


    def step_slow(self, x, last_b, history):
        if len(history) <= self.window:
            return last_b
        else:
            # init
            window = self.window
            indices = []
            m = len(x)

            # calculate correlation with predecesors
            X_t = history.iloc[-window:].values.flatten()
            for i in range(window, len(history)):
                X_i = history.ix[i-window:i-1].values.flatten()
                if np.corrcoef(X_t, X_i)[0,1] >= self.rho:
                    indices.append(i)

            # calculate optimal portfolio
            C = history.ix[indices, :]

            if C.shape[0] == 0:
                b = np.ones(m) / float(m)
            else:
                b = self.optimal_weights(C)

            return b


    def step_fast(self, x, last_b):
        # iterate time
        self.t += 1

        if self.t < self.window:
            return last_b
        else:
            # init
            window = self.window
            m = len(x)

            X_t = self.X_flat.ix[self.t]
            X_i = self.X_flat.iloc[window-1 : self.t]
            c = X_i.apply(lambda r: np.corrcoef(r.values, X_t.values)[0,1], axis=1)

            C = self.X.ix[c.index[c >= self.rho] + 1]

            if C.shape[0] == 0:
                b = np.ones(m) / float(m)
            else:
                b = self.optimal_weights(C)

            return b

    def optimal_weights(self, X):
        X = np.mat(X)

        n,m = X.shape
        P = 2 * matrix(X.T * X)
        q = -3 * matrix(np.ones((1,n)) * X).T

        G = matrix(-np.eye(m))
        h = matrix(np.zeros(m))
        A = matrix(np.ones(m)).T
        b = matrix(1.)

        sol = solvers.qp(P, q, G, h, A, b)
        return np.squeeze(sol['x'])



# use case
if __name__ == '__main__':
    tools.quickrun(CORN())
