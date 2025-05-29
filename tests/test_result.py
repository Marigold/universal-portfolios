from universal import algos


def test_turnover_and_fees(S):
    algo = algos.BAH()
    result = algo.run(S)
    result.fee = 0.01
    assert abs(result.turnover) < 1e-10
    assert abs(result.fees.sum().sum()) < 1e-10
