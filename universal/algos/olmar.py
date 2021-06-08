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

    def __init__(self, window=5, eps=10):
        """
        :param window: Lookback window.
        :param eps: Constraint on return for new weights on last price (average of prices).
            x * w >= eps for new weights w.
        """

        super(OLMAR, self).__init__(min_history=window)

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
        # calculate return prediction
        x_pred = self.predict(x, history.iloc[-self.window:])
        
        # Update the weights
        b = self.update_olmar(last_b, x_pred, self.eps)
        b = self.update_tco(b, x_pred)
        return b


    @staticmethod
    def predict(x, history):
        """ Predict returns on next day. """
        return (history / x).mean()

    
    @staticmethod
    def update_olmar(b, x, eps):
        """ Update portfolio weights to satisfy constraint b * x >= eps
        and minimize distance to previous weights. """
        x_mean = np.mean(x)
        lam = max(0., (eps - np.dot(b, x)) / np.linalg.norm(x - x_mean)**2)

        # limit lambda to avoid numerical problems
        lam = min(100000, lam)

        # update portfolio
        b = b + lam * (x - x_mean)

        # project it onto simplex
        return tools.simplex_proj(b)

    def update_tco(self, b, x_pred):
        """
        Transaction Costs Optimization
        Paper : https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?referer=&httpsredir=1&article=4761&context=sis_research
        """

        trx_fee_pct = 0.1
        n = 10
        lambd = 10*trx_fee_pct

        # last price adjusted weights
        updated_b = np.multiply(b, x_pred) / np.dot(b, x_pred)

        # Calculate variables
        vt   = x_pred / np.dot(updated_b, x_pred)
        v_t_ = np.dot(1, vt) / self.window

        # Update portfolio
        b_1 = n * (vt - np.dot(v_t_, 1))
        b_  = b_1 + np.sign(b_1)*np.maximum(np.zeros(len(b_1)), np.abs(b_1) - lambd)

        # project it onto simplex
        return tools.simplex_proj(y=b_)

    
if __name__ == '__main__':
    tools.quickrun(OLMAR())

