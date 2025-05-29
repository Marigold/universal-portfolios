import pytest

from universal import algos

EPS = 1e-10


@pytest.mark.parametrize(
    "algo_class",
    [
        algos.CRP,
        algos.RMR,
        algos.OLMAR,
        algos.PAMR,
    ],
)
def test_bias(algo_class, S):
    """Test forward bias of algo. Test on a portion of given data set, then add several
    data points and see if weights has changed.
    """
    m = 10
    B1 = algo_class().run(S.iloc[:-m]).B
    B2 = algo_class().run(S).B

    assert (B1 == B2.iloc[:-m]).all().all()


# BAH
def test_bah(S):
    """There should be no fees for BAH."""
    FEE = 0.01
    result = algos.BAH().run(S)
    wealth_no_fees = result.total_wealth
    result.fee = FEE
    wealth_with_fees = result.total_wealth

    assert abs(wealth_no_fees - wealth_with_fees) < EPS


# CRP
def test_crp(S):
    """Make sure that equity of a portfolio [1,0,...,0] with NaN values
    is the same as asset itself."""
    b = [1.0] + [0.0] * (len(S.columns) - 1)
    result = algos.CRP(b).run(S)

    assert abs(result.total_wealth - S[S.columns[0]].iloc[-1]) < EPS


def test_tco1(S):
    """Zero turnover with extremely high fees."""
    result = algos.TCO1(eta=1, trx_fee_pct=1e6).run(S)
    assert abs(result.turnover) < 1e-8
