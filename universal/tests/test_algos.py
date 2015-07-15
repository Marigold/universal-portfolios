import pytest
from universal import algos
from .. import tools


EPS = 1e-10


@pytest.fixture(scope="module")
def S():
    """ Random portfolio for testing. """
    return tools.random_portfolio(n=1000, k=3, mu=0., sd=0.01)


@pytest.mark.parametrize("algo_class", [
    algos.CRP,
    algos.RMR,
    algos.OLMAR,
    algos.PAMR,
])
def test_bias(algo_class, S):
    """ Test forward bias of algo. Test on a portion of given data set, then add several
    data points and see if weights has changed. """
    m = 10
    B1 = algo_class().run(S.iloc[:-m]).B
    B2 = algo_class().run(S).B

    assert (B1 == B2.iloc[:-m]).all().all()


# BAH
def test_bah(S):
    """ Fees for BAH should be equal to 1 * fee. """
    FEE = 0.01
    result = algos.BAH().run(S)
    wealth_no_fees = result.total_wealth
    result.fee = FEE
    wealth_with_fees = result.total_wealth

    assert abs(wealth_no_fees * (1 - FEE) - wealth_with_fees) < EPS


# CRP
def test_crp(S):
    """ Make sure that equity of a portfolio [1,0,...,0] with NaN values
    is the same as asset itself. """
    b = [1.] + [0.] * (len(S.columns) - 1)
    result = algos.CRP(b).run(S)

    assert abs(result.total_wealth - S[S.columns[0]].iget(-1)) < EPS
