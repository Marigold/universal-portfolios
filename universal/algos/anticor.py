import warnings

import numpy as np

from .. import tools
from ..algo import Algo


class Anticor(Algo):
    """Anticor (anti-correlation) is a heuristic portfolio selection algorithm.
    It adopts the consistency of positive lagged cross-correlation and negative
    autocorrelation to adjust the portfolio. Eventhough it has no known bounds and
    hence is not considered to be universal, it has very strong empirical results.

    It has implemented C version in scipy.weave to improve performance (around 10x speed up).
    Another option is to use Numba.

    Reference:
        A. Borodin, R. El-Yaniv, and V. Gogan.  Can we learn to beat the best stock, 2005.
        http://www.cs.technion.ac.il/~rani/el-yaniv-papers/BorodinEG03.pdf
    """

    def __init__(self, window=30, c_version=True):
        """
        :param window: Window parameter.
        :param c_version: Use c_version, up to 10x speed-up.
        """
        super().__init__()
        self.window = window
        self.c_version = c_version

    def weights(self, X):
        window = self.window
        port = X
        n, m = port.shape
        weights = 1.0 / m * np.ones(port.shape)

        CORR, EX = tools.rolling_corr(port, port.shift(window), window=window)

        if self.c_version:
            try:
                from scipy import weave
            except ImportError:
                warnings.warn(
                    "scipy.weave is not available in python3, falling back to python version"
                )
                self.c_version = False

        if self.c_version is False:
            for t in range(n - 1):
                M = CORR[t, :, :]
                mu = EX[t, :]

                # claim[i,j] is claim from stock i to j
                claim = np.zeros((m, m))

                for i in range(m):
                    for j in range(m):
                        if i == j:
                            continue

                        if mu[i] > mu[j] and M[i, j] > 0:
                            claim[i, j] += M[i, j]
                            # autocorrelation
                            if M[i, i] < 0:
                                claim[i, j] += abs(M[i, i])
                            if M[j, j] < 0:
                                claim[i, j] += abs(M[j, j])

                # calculate transfer
                claim = np.nan_to_num(claim) #correlation can be nan
                transfer = claim * 0.0
                for i in range(m):
                    total_claim = sum(claim[i, :])
                    if total_claim != 0 and not np.isnan(total_claim).any(): #transfer should never be nan
                        transfer[i, :] = weights[t, i] * claim[i, :] / total_claim

                # update weights
                weights[t + 1, :] = (
                    weights[t, :] + np.sum(transfer, axis=0) - np.sum(transfer, axis=1)
                )

        else:

            def get_weights_c(c, mu, w):
                code = """
                int t,i,j;
                float claim [Nc[1]] [Nc[1]];
                float transfer [Nc[1]] [Nc[1]];

                for (t=0; t<Nc[0]-1; t++) {

                    for (i=0; i<Nc[1]; i++) {
                        for (j=0; j<Nc[1]; j++) {
                            claim[i][j] = 0.;
                            transfer[i][j] = 0.;
                        }
                    }

                    for (i=0; i<Nc[1]; i++) {
                        for (j=0; j<Nc[1]; j++) {
                            if(i != j){
                                if(MU2(t,i) > MU2(t,j)  && C3(t,i,j) > 0){
                                    claim[i][j] += C3(t,i,j);
                                    if(C3(t,i,i) < 0)
                                        claim[i][j] -= C3(t,i,i);
                                    if(C3(t,j,j) < 0)
                                        claim[i][j] -= C3(t,j,j);
                                }
                            }
                        }
                    }

                    for (i=0; i<Nc[1]; i++) {
                        float total_claim=0.;
                        for (j=0; j<Nc[1]; j++) {
                            total_claim += claim[i][j];
                        }
                        if(total_claim != 0){
                            for (j=0; j<Nc[1]; j++) {
                                transfer[i][j] = W2(t,i) * claim[i][j] / total_claim;
                            }
                        }

                    }

                    for (i=0; i<Nc[1]; i++) {
                        W2(t+1,i) = W2(t,i);
                        for (j=0; j<Nc[1]; j++) {
                            W2(t+1,i) += transfer[j][i] - transfer[i][j];
                        }
                    }
                }
                """
                return weave.inline(code, ["c", "mu", "w"])

            get_weights_c(CORR, EX, weights)

        return weights


if __name__ == "__main__":
    tools.quickrun(Anticor())
