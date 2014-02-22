# -*- coding: utf-8 -*-
import unittest
from universal.algos import CRP, BAH
import universal.tools as tools

"""
Unit tests for algos. Used for testing forward bias, etc.
TODO: complete the tests before you will regret it...
"""

class TestCRP(unittest.TestCase):

    def setUp(self):
        self.random_portfolio = tools.random_portfolio(n=1000, k=3, mu=0., sd=0.01)

    def test_equity(self):
        """ Make sure that equity of a portfolio [1,0,...,0] with NaN values
        is the same as asset itself. """
        S = self.random_portfolio
        b = [1.] + [0.] * (len(S.columns) - 1)
        result = CRP(b).run(S)

        self.assertAlmostEqual(result.total_wealth,
                               S[S.columns[0]].iget(-1), 10)

    def test_bias(self):
        """ Test forward bias of algo. Test on a portion of given data set, then add several
        data points and see if weights has changed. """
        S = self.random_portfolio
        k = 10
        B1 = CRP().run(S.iloc[:-k]).B
        B2 = CRP().run(S).B

        self.assertTrue((B1 == B2.iloc[:-k]).all().all())

    def test_fees(self):
        """ Fees for BAH should be equal to 1 * fee. """
        FEE = 0.01
        S = self.random_portfolio
        result = BAH().run(S)
        wealth_no_fees = result.total_wealth
        result.fee = FEE
        wealth_with_fees = result.total_wealth

        self.assertAlmostEqual(wealth_no_fees * (1 - FEE), wealth_with_fees, 10)



if __name__ == '__main__':
    unittest.main()