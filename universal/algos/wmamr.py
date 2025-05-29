from universal.algos.pamr import PAMR

from .. import tools


class WMAMR(PAMR):
    """Weighted Moving Average Passive Aggressive Algorithm for Online Portfolio Selection.
    It is just a combination of OLMAR and PAMR, where we use mean of past returns to predict
    next day's return.

    Reference:
        Li Gao, Weiguo Zhang
        Weighted Moving Averag Passive Aggressive Algorithm for Online Portfolio Selection, 2013.
        http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=6643896
    """

    PRICE_TYPE = "ratio"

    def __init__(self, window=5, **kwargs):
        """
        :param w: Windows length for moving average.
        :param kwargs: Additional arguments for PAMR.
        """
        super().__init__(**kwargs)

        if window < 1:
            raise ValueError("window parameter must be >=1")
        self.window = window

    def step(self, x, last_b, history):
        xx = history[-self.window :].mean()
        # calculate return prediction
        b = self.update(last_b, xx, self.eps, self.C)
        return b


# use case
if __name__ == "__main__":
    tools.quickrun(WMAMR())
