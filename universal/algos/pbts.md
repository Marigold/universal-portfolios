# PBTS: Portfolio Blending via Thompson Sampling

[`pbts.py`](pbts.py) holds a blend of the minimum-variance portfolio (MV) and a benchmark, either the equally weighted portfolio (EW) or the value-weighted one (VW):

```math
\omega_k = \delta_k \, \omega^{MV}_k + (1 - \delta_k)\, \omega^{EW (VW)}_k .
```

The blending coefficient $`\delta_k`$ is re-estimated every period from the last $`\tau`$ periods, framed as a Bernoulli bandit and solved with Thompson sampling. With EW this is the paper's TS-EM portfolio, and with VW it is TS-VM.

Reference: W. Shen, J. Wang, [Portfolio Blending via Thompson Sampling](https://www.ijcai.org/Proceedings/16/Papers/283.pdf), IJCAI 2016. Equation numbers below refer to that paper.

## What it is good for

PBTS aims at risk-adjusted returns, not wealth. It never moves further from its benchmark than the minimum-variance portfolio is, and it leans towards MV after periods in which MV did better. So expect it to sit between uniform CRP and a long-only minimum-variance portfolio in both return and volatility. The paper reports higher Sharpe ratios than EW, VW, MV and three other blends on monthly Fama French portfolios and two weekly datasets, with transaction costs of 50 basis points. It makes no regret claim.

On the bundled daily datasets with the default $`\tau = 120`$ days, the results are below. MV is a long-only minimum-variance portfolio on the same window, which is PBTS with $`\delta = 1`$. `MPT(method="variance")` would be the natural comparison, but it currently fails with `AttributeError: 'MPT' object has no attribute 'allow_cash'`. Volatility is annualised. Wealth is before fees.

| Dataset | Strategy | Wealth | Sharpe | Volatility | Max drawdown | Mean $`\delta^*`$ |
|---|---|---:|---:|---:|---:|---:|
| NYSE(O) | CRP | 26.68 | 1.16 | 13.4% | 36.6% | |
| | MV | 14.37 | 1.11 | 11.3% | 46.3% | |
| | PBTS EW | 19.50 | 1.18 | 11.9% | 41.5% | 0.49 |
| | PBTS VW | 14.63 | 1.03 | 12.4% | 44.1% | 0.49 |
| DJIA | CRP | 0.81 | -0.28 | 25.5% | 37.8% | |
| | MV | 0.71 | -0.77 | 19.7% | 35.2% | |
| | PBTS EW | 0.75 | -0.58 | 21.2% | 35.9% | 0.55 |
| | PBTS VW | 0.72 | -0.68 | 20.8% | 36.4% | 0.54 |
| TSE | CRP | 1.57 | 0.76 | 13.0% | 33.7% | |
| | MV | 1.73 | 1.19 | 9.6% | 23.5% | |
| | PBTS EW | 1.64 | 0.98 | 10.6% | 27.6% | 0.49 |
| | PBTS VW | 1.65 | 0.98 | 10.8% | 26.9% | 0.46 |
| SP500 | CRP | 1.64 | 0.55 | 22.0% | 31.1% | |
| | MV | 1.16 | 0.26 | 18.0% | 32.1% | |
| | PBTS EW | 1.38 | 0.43 | 19.1% | 29.9% | 0.50 |
| | PBTS VW | 1.26 | 0.33 | 19.9% | 37.1% | 0.50 |

Three things stand out:
- **It is close to a fixed half-and-half blend.** On daily data MV beats the benchmark on about half the days, so $`\delta^*`$ averages about 0.5 on every dataset. PBTS then sits between MV and its benchmark in volatility and Sharpe ratio, but rarely beats both.
- **It lowers volatility.** Volatility is below CRP's on all four datasets. The Sharpe ratio is above CRP's only on NYSE(O), slightly, and on TSE.
- **It trades a lot.** Turnover is about three times CRP's. At a fee of 0.2% per unit traded, PBTS EW's wealth on NYSE(O) falls from 19.50 to 11.55, while CRP's falls from 26.68 to 23.41.

The paper's monthly and weekly results may not carry over to daily rebalancing.

## Setup

There are $`n`$ assets and rebalancing periods $`k = 1, \dots, m`$. $`R_k`$ is the vector of gross returns $`S_{k,i}/S_{k-1,i}`$ over period $`k`$. The three basis portfolios are:

```math
\omega^{EW}_k = \tfrac{1}{n}\mathbf{1} \quad (2), \qquad
\omega^{VW}_k = \frac{\omega_{k-1} \circ R_{k-1}}{\omega_{k-1}^\top R_{k-1}} \quad (3), \qquad
\omega^{MV}_k = \arg\min_{\omega^\top \mathbf{1} = 1} \omega^\top \Sigma_k \omega = \frac{\Sigma_k^{-1}\mathbf{1}}{\mathbf{1}^\top \Sigma_k^{-1}\mathbf{1}} \quad (4),
```

where $`\Sigma_k`$ is the covariance of $`R_{k-\tau}, \dots, R_{k-1}`$.

## Algorithm 1

For each period $`k`$:

1. Compute $`\omega^{EW}_k`$, $`\omega^{VW}_k`$ and $`\omega^{MV}_k`$.
2. Start from a $`\text{Beta}(a, b)`$ prior with $`a = b = 1`$.
3. For each of the $`\tau`$ periods $`j`$ in the window:
   - Take the benchmark coefficient $`\bar\delta_j = a/(a+b)`$, the Beta mean. Build the benchmark blend $`\bar\omega_j = \bar\delta_j \omega^{MV}_k + (1-\bar\delta_j)\omega^{EW(VW)}_k`$ (eq. 7).
   - Sample $`\tilde\delta_j \sim \text{Beta}(a, b)`$ and build the test blend $`\tilde\omega_j`$ the same way (eq. 8).
   - Decide the test with eq. (9). It is a success if $`R^\top\tilde\omega > R^\top\bar\omega`$ and $`\tilde\delta > \bar\delta`$, or if $`R^\top\tilde\omega < R^\top\bar\omega`$ and $`\tilde\delta < \bar\delta`$. Otherwise it is a failure.
   - On a success $`a \gets a+1`$, and on a failure $`b \gets b+1`$. Equal returns update nothing, and a sample equal to the mean is redrawn (footnote 1).
4. Set $`\delta^*_k = (1+a_\tau)/(1+a_\tau+1+b_\tau)`$, the posterior mean (eq. 10). Hold $`\delta^*_k \omega^{MV}_k + (1-\delta^*_k)\omega^{EW(VW)}_k`$ (eq. 11).

The paper uses $`\tau = 120`$ months or $`\tau = 200`$ weeks.

## The sample cancels out

The two blends in a test differ only in their coefficient:

```math
\tilde\omega_j - \bar\omega_j = (\tilde\delta_j - \bar\delta_j)\,(\omega^{MV}_k - \omega^{EW(VW)}_k),
\qquad
R_j^\top\tilde\omega_j - R_j^\top\bar\omega_j = (\tilde\delta_j - \bar\delta_j)\; R_j^\top(\omega^{MV}_k - \omega^{EW(VW)}_k).
```

Eq. (9) calls the test a success when the return difference has the same sign as $`\tilde\delta_j - \bar\delta_j`$. Write $`e_j = R_j^\top(\omega^{MV}_k - \omega^{EW(VW)}_k)`$ for MV's edge over the benchmark in period $`j`$. The return difference is $`(\tilde\delta_j - \bar\delta_j) e_j`$, so its sign matches the sign of $`\tilde\delta_j - \bar\delta_j`$ exactly when $`e_j > 0`$. So for any $`\tilde\delta_j \ne \bar\delta_j`$:

- success $`\iff e_j > 0`$, meaning MV beat the benchmark in period $`j`$;
- failure $`\iff e_j < 0`$;
- no update $`\iff e_j = 0`$.

The samples never change the outcome. $`\delta^*_k`$ is the Laplace-smoothed share of the window in which MV beat the benchmark:

```math
\delta^*_k = \frac{1 + \#\{j : e_j > 0\}}{2 + \#\{j : e_j \ne 0\}} .
```

The algorithm is therefore deterministic given the data, and the exploration that Thompson sampling usually provides doesn't happen here. The code still follows Algorithm 1 with real samples, so it reads like the paper. `test_sampled_delta_cancels_out` checks the identity, and `test_matches_transcription_of_algorithm_1` checks the whole algorithm against an independent loop that uses this closed form.

## Parameters

- `benchmark`: `"EW"` (TS-EM) or `"VW"` (TS-VM).
- `tau`: the window, used both for $`\Sigma_k`$ and for the Bernoulli tests. Larger windows make $`\delta^*_k`$ move more slowly. If MV won all $`\tau`$ periods, $`\delta^* = (\tau+1)/(\tau+2)`$, and if it lost all of them, $`\delta^* = 1/(\tau+2)`$, so $`\delta^*`$ never reaches 0 or 1.
- `cov_estimator`: how $`\Sigma_k`$ is estimated (see departures).
- `seed`: only decides how exact ties of $`\tilde\delta_j`$ with $`\bar\delta_j`$ are redrawn, which happens with probability zero.

## Departures from the paper

- **Long-only MV.** Eq. (4) is the unconstrained minimum-variance portfolio, which can go short. Following the library maintainer's preference, `pbts.py` adds $`\omega \ge 0`$ and solves the quadratic program with cvxopt, as `MPT(method="variance")` does. With this constraint the blend is long-only too.
- **Covariance.** The paper estimates $`\Sigma_k`$ with a factor model (Fan et al., 2008). The default here is the empirical covariance of the window. `"ledoit-wolf"`, `"oas"` or any scikit-learn covariance estimator can be passed instead.
- **VW without market capitalisations.** Eq. (3) lets the previous value-weighted portfolio drift with prices. The bundled datasets have no market capitalisations, so VW starts from uniform weights and is buy and hold from then on.
- **Warm-up.** The paper starts with $`\tau`$ periods of training data. Here, until $`\tau`$ periods have been seen, PBTS holds the benchmark: EW, or VW, which is buy and hold.
- **Ties within rounding.** Footnote 1 skips the update when the two blends earn exactly the same. Weights that sum to one only up to floating-point rounding would otherwise decide a test by rounding error, for example on the first period's "no change" returns. So the code treats return differences within $`10^{-12}`$ as ties.

## How the code maps to this

| Paper | Code |
|---|---|
| $`R_k`$ | `x` in `step`; the window is `self.returns` |
| $`\omega^{EW}_k`$, $`\omega^{VW}_k`$ | `benchmark` in `step`; VW is `self.vw` |
| $`\Sigma_k`$, $`\omega^{MV}_k`$ | `min_variance`, `long_only_min_variance` |
| $`a, b`$, $`\bar\delta_j`$, $`\tilde\delta_j`$ | `a`, `b`, `delta_bar`, `delta_tilde` in `blending_coefficient` |
| eq. (9) | the sign test on `gain` |
| $`\delta^*_k`$, eq. (10) | return value of `blending_coefficient` |
| eq. (11) | return value of `step` |

**Timing.** `step` is called after period $`t`$ with that period's returns `x`. It appends `x` to the window first, then computes the weights held in period $`t+1`$. So the window for those weights is periods $`t-\tau+1, \dots, t`$, which is the paper's $`R_{k-\tau}, \dots, R_{k-1}`$ with $`k = t+1`$. `test_uses_the_latest_return` fails if the weights lag by one period. With `frequency > 1`, each step gets the return accumulated since the previous step, and the window holds those, so $`\tau`$ counts rebalancing periods as in the paper.

## Limitations

- The "Thompson sampling" is decorative: as shown above, the samples never affect the decision. Whatever value PBTS adds comes from the frequency rule $`\delta^*_k`$, not from exploration.
- $`\delta^*_k`$ counts wins and ignores their size. A window where MV wins many periods by a little and loses a few by a lot still tilts towards MV.
- With the default empirical covariance and $`\tau`$ smaller than the number of assets, $`\Sigma_k`$ is singular. The long-only problem is still well defined, but the solution can be unstable. A shrinkage estimator such as `"ledoit-wolf"` helps.
- The paper's results are on monthly and weekly data with $`\tau`$ of 120 or 200 rebalancing periods. On the bundled daily datasets, the default $`\tau = 120`$ is about half a year.
