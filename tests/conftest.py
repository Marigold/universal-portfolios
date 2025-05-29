import pytest

from universal import tools


@pytest.fixture(scope="module")
def S():
    """Random portfolio for testing."""
    return tools.random_portfolio(n=100, k=3, mu=0.0, sd=0.01)
