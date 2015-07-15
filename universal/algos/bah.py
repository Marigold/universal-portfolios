from ..algo import Algo
from .. import tools
import numpy as np


class BAH(Algo):
    """ Bay and hold strategy. Buy equal amount of each stock in the beginning and hold them
    forever.  """

    PRICE_TYPE = 'raw'

    def __init__(self, b=None):
        """
        :params b: Portfolio weights at start. Default are uniform.
        """
        super(BAH, self).__init__()
        self.b = b

    def weights(self, S):
        """ Weights function optimized for performance. """
        b = np.ones(S.shape[1]) / S.shape[1] if self.b is None else self.b

        # weights are proportional to price times initial weights
        w = S * b

        # normalize
        w = w.div(w.sum(axis=1), axis=0)

        # shift
        w = w.shift(1)
        w.ix[0] = 1./S.shape[1]

        return w


if __name__ == '__main__':
    tools.quickrun(BAH())
