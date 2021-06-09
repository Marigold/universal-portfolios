from ..algo import Algo
import numpy as np
import pandas as pd
from .. import tools


class OLMAR(Algo):
    """ On-Line Portfolio Selection with Moving Average Reversion

    Reference:
        B. Li and S. C. H. Hoi.
        On-line portfolio selection with moving average reversion, 2012.
        http://icml.cc/2012/papers/168.pdf
    """

    PRICE_TYPE = 'raw'
    REPLACE_MISSING = True

    def __init__(self, window=5, eps=10, **kwargs):
        """
        :param window: Lookback window.
        :param eps: Constraint on return for new weights on last price (average of prices).
            x * w >= eps for new weights w.
        """

        super(OLMAR, self).__init__(min_history = window,
                                    trx_fee_pct = kwargs.get('trx_fee_pct', 0),
                                    n = kwargs.get('n', 10),)

        # input check
        if window < 2:
            raise ValueError('window parameter must be >=3')
        if eps < 1:
            raise ValueError('epsilon parameter must be >=1')

        self.window = window
        self.eps = eps


    def init_weights(self, columns):
        m = len(columns)
        return np.ones(m) / m


    def step(self, x, last_b, history):
        """
        Called at each time step.
        """
        # calculate return prediction
        x_pred = self.predict(x, history.iloc[-self.window:])
        
        # Update the weights
        b = self.update_olmar(last_b, x_pred, self.eps)

        return b


    @staticmethod
    def predict(x, history):
        """ Predict returns on next day. """
        return (history / x).mean()

    
    @staticmethod
    def update_olmar(b, x, eps):
        """ 
        Update portfolio weights to satisfy constraint b * x >= eps
        and minimize distance to previous weights.
        """
        x_mean = np.mean(x)
        lam = max(0., (eps - np.dot(b, x)) / np.linalg.norm(x - x_mean)**2)

        # limit lambda to avoid numerical problems
        lam = min(100000, lam)

        # update portfolio
        b = b + lam * (x - x_mean)

        # project it onto simplex
        return tools.simplex_proj(b)

    
if __name__ == '__main__':
    tools.quickrun(OLMAR())

