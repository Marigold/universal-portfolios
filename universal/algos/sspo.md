# SSPO

Short-term Sparse Portfolio Optimization (SSPO) puts almost all of its wealth into the asset or two whose price sits furthest below its recent peak, betting that it climbs back up. [`sspo.py`](sspo.py) implements it as the paper's Algorithm 1, which solves an $`\ell_1`$-regularized linear program with ADMM every period and projects the scaled solution onto the simplex.

Reference: Z.-R. Lai, P.-Y. Yang, L. Fang, X. Wu, [Short-term Sparse Portfolio Optimization Based on Alternating Direction Method of Multipliers](https://jmlr.org/papers/v19/17-558.html), JMLR 19(63) (2018). The authors' MATLAB code is at [laizhr/SSPO](https://github.com/laizhr/SSPO). Equation, table and section numbers below refer to the paper.

## What it is good for

SSPO is an aggressive daily strategy. With the default parameters it holds about one asset at a time (1.01 on average on every bundled dataset) and moves about half of its wealth every period. It is not a diversifier, and its returns carry the risk of a single stock.

On the classic benchmark datasets it does very well without fees: far above uniform CRP, above OLMAR on all four datasets below, and above RPRT (by the same authors) on DJIA only. Because it trades less than OLMAR and RPRT, it keeps more of its wealth under proportional transaction costs (Section 4.8 of the paper makes the same point), but a 0.5% round-trip cost still takes most of it, and on SP500 it then ends below uniform CRP. Strategies that buy recent losers are known to do unusually well on these old datasets, so treat the numbers as a comparison between strategies, not as a forecast.

## Setup

There are $`d`$ assets with prices $`p_t`$ and price relatives $`x_t = p_t / p_{t-1}`$. The portfolio $`\hat b_t`$ lies on the simplex $`\Delta_d = \{b \ge 0 : \mathbf 1^\top b = 1\}`$ and is held during period $`t`$, so wealth grows as $`S_t = S_{t-1} \, \hat b_t^\top x_t`$ (eq. 1).

## Increasing potential

SSPO scores each asset by how far its current price is below the highest price of the last $`w`$ periods (eqs. 10 and 11):

```math
p^{(i)}_{MAX} = \max_{0 \le k \le w-1} p^{(i)}_{t-k}, \qquad R_t = 1.1 \log\left(\frac{p_{MAX}}{p_t}\right) + 1, \qquad \varphi_t = -R_t .
```

$`R_t \ge 1`$, with $`R^{(i)}_t = 1`$ when asset $`i`$ is at its peak. The paper presents this as exploiting overreaction rather than mean reversion, but in practice it buys the assets that have fallen the most within the window.

## Sparse portfolio model

The next portfolio maximizes the increasing potential with an $`\ell_1`$ penalty and the self-financing constraint (eq. 12):

```math
b_{t+1} = \min_b \; b^\top \varphi_t + \lambda \lVert b \rVert_1 \quad \text{s.t.} \quad \mathbf 1^\top b = 1 .
```

ADMM splits $`b`$ from an auxiliary $`g`$ that carries the $`\ell_1`$ term and adds a dual variable $`\rho`$ for the constraint, giving the augmented Lagrangian (eq. 13)

```math
L(b, g, \rho) = b^\top \varphi_t + \frac{\lambda}{2\gamma} \lVert b - g \rVert^2 + \lambda \lVert g \rVert_1 + \frac{\eta}{2} (\mathbf 1^\top b - 1)^2 + \rho (\mathbf 1^\top b - 1).
```

## Algorithm 1

Each period, starting from $`b^{(1)} = g^{(1)} = \hat b_t`$ (the weights currently held) and $`\rho^{(1)} = 0`$, repeat

```math
\begin{aligned}
b^{(o+1)} &= \left(\frac{\lambda}{\gamma} I + \eta \mathbf 1 \mathbf 1^\top\right)^{-1} \left(\frac{\lambda}{\gamma} g^{(o)} + (\eta - \rho^{(o)}) \mathbf 1 - \varphi_t\right) && \text{(eq. 29)} \\
g^{(o+1)} &= \mathrm{sign}(b^{(o+1)}) \otimes \left(\mathrm{abs}(b^{(o+1)}) - \gamma \mathbf 1\right)_+ && \text{(eq. 32)} \\
\rho^{(o+1)} &= \rho^{(o)} + \eta (\mathbf 1^\top b^{(o+1)} - 1) && \text{(eq. 27)}
\end{aligned}
```

until $`|\mathbf 1^\top b^{(o)} - 1| < \epsilon`$ or $`10^4`$ iterations, then normalize (eq. 33):

```math
\hat b_{t+1} = \arg\min_{b \in \Delta_d} \lVert b - \zeta b^{(o)} \rVert^2 .
```

The b-step is a ridge-like solve. With $`\tau = \lambda / \gamma`$, the Sherman–Morrison formula gives

```math
\left(\tau I + \eta \mathbf 1 \mathbf 1^\top\right)^{-1} = \frac{1}{\tau} \left(I - \frac{\eta}{\tau + \eta d} \mathbf 1 \mathbf 1^\top\right),
```

so the code computes it in $`O(d)`$ without a matrix solve. The part that depends on $`\varphi_t`$ is fixed for the period, and the rest is $`g^{(o)}`$ plus a multiple of $`\mathbf 1`$.

## Parameters

Defaults are the paper's (Section 4.1) and the authors' code's: $`w = 5`$, $`\lambda = 0.5`$, $`\gamma = 0.01`$, $`\eta = 0.005`$, $`\zeta = 500`$, $`\epsilon = 10^{-4}`$, at most $`10^4`$ iterations.

| parameter | code | effect |
|---|---|---|
| $`w`$ | `window` | lookback for the peak price |
| $`\lambda`$ | `lam` | strength of the $`\ell_1`$ penalty, i.e. sparsity of $`b^{(o)}`$ |
| $`\gamma`$ | `gamma` | soft-threshold step; how closely $`g`$ tracks $`b`$ |
| $`\eta`$ | `eta` | penalty on $`\mathbf 1^\top b \ne 1`$ and the dual step size |
| $`\zeta`$ | `zeta` | scale before the projection |
| $`\epsilon`$ | `tol` | stopping tolerance on the constraint |
| | `max_iter` | iteration cap per period |

$`\zeta`$ decides how concentrated the final portfolio is. As $`\zeta \to 0`$ the projection of $`\zeta b^{(o)}`$ becomes the uniform portfolio, so SSPO becomes uniform [CRP](crp.py). With $`\zeta = 500`$ the projection keeps only the largest one or two entries of $`b^{(o)}`$. The paper tunes each parameter on the five datasets with the others fixed (Figures 3 to 6).

## Guarantees

The paper proves that the augmented Lagrangian (13) has a saddle point (Section 3.3.1, eq. 14), which justifies the ADMM iterations, and that the b-step has a unique minimizer (Proposition 1). There is no regret or wealth guarantee. The case for SSPO is empirical: the final wealths in Table 3, the sparsity in Table 1, and the transaction-cost and running-time experiments in Sections 4.8 and 4.9.

The sparsity of eq. (34), the share of non-maximal entries of $`b^{(o)}`$ that are at most 10% of the maximal one, averages 91.84% on DJIA here against 91.91% in Table 1, 91.36% on SP500 (91.36%) and 94.49% on TSE (94.50%).

## First days

The paper doesn't say what happens before a full window exists. The authors' `SSPO_run.m` scores assets by the last price relative, $`R_t = 1.1 \log x_t + 1`$, while it has at most $`w`$ closing prices, and switches to the peak rule once it has $`w + 1`$. During those first days SSPO follows recent winners, not recent losers. The code does the same, counting the prices in `history`.

The bundled CSVs are cumulative products of the price relatives, so their first row is the price after day 1, not a base price. `Algo.run` treats the first row as the base, so it never sees the first price relative and holds uniform weights for one more day. Fed the relatives the library sees ($`x_1`$ replaced by 1), the authors' code gives the same weights as `SSPO().run(S)` to within $`10^{-11}`$ on all five datasets. Fed the original relatives, it reproduces Table 3 to every printed digit. The only difference is therefore that missing first day, and because SSPO holds a single asset and warm-starts ADMM from the previous weights, that one day moves the final wealth by −5.6% to +1.8%:

| dataset | Table 3 | authors' code | `SSPO().run(S)` |
|---|---|---|---|
| NYSE(O) | 1.06E+18 | 1.0586E+18 | 1.0149E+18 |
| NYSE(N) | 1.62E+9 | 1.6203E+9 | 1.5291E+9 |
| DJIA | 3.68 | 3.677 | 3.714 |
| SP500 | 16.97 | 16.968 | 17.277 |
| TSE | 364.94 | 364.944 | 349.465 |

Prepending a row of ones to the CSV restores the first day but gives the code one more price than the authors' `data_close` holds, which starts at the price after day 1. The switch to the peak rule then comes one day earlier than in the authors' code.

## How the code maps to this

| math | code |
|---|---|
| $`p_{t-k}`$, $`0 \le k \le w-1`$ | `history.iloc[-self.window:]`; with `PRICE_TYPE = "raw"`, `history` is prices normalized to start at 1 |
| $`\varphi_t = -R_t`$ | `phi` in `step` |
| $`\hat b_t`$, the warm start | `last_b` |
| $`b^{(o)}`$ before the projection | the return value of `update(last_b, phi)` |
| $`\tau = \lambda / \gamma`$ | `tau` |
| $`(\tau I + \eta \mathbf 1 \mathbf 1^\top)^{-1}(-\varphi_t)`$ | `u` |
| $`\rho`$ | `rho` |
| eq. (33) | `tools.simplex_proj(self.zeta * b)` |

`step` is called after period $`t`$, with `history` holding the prices up to and including $`p_t`$. It returns $`\hat b_{t+1}`$, which the framework holds during period $`t+1`$. SSPO keeps no state between steps: everything is recomputed from `history`, and the warm start is the `last_b` that `Algo` passes in, so `next_weights` and `frequency > 1` work without an `init_step`.

## Departures from the authors' code

- `simplex_projection_selfnorm2.m` divides the vector by 10 while its largest entry exceeds $`10^6`$ before projecting. The projection isn't scale invariant, so this would change the weights when it triggers. It is a numerical guard from the projection routine the authors borrowed, not part of the paper, and it never triggers on the bundled data: the largest $`|\zeta b^{(o)}|`$ is about 2,500 on TSE and 1,300 on DJIA. The code leaves it out.
- The b-step uses Sherman–Morrison instead of the matrix solve. It agrees with the dense solve of eq. (29) to rounding.

## Limitations

- **Concentration.** About one asset at a time, so single-stock risk.
- **Turnover.** About half the wealth changes hands each period. Set `result.fee` before trusting a comparison: the library's `fee` is charged per unit traded, so the paper's round-trip rate $`\nu`$ of eq. (40) is `fee = nu / 2`.
- **Sensitivity.** One different early day changes the final wealth by several percent (see [First days](#first-days)), so differences between SSPO and other strategies at the third significant digit mean little.
- **The stopping rule checks only the constraint.** ADMM can stop as soon as $`\mathbf 1^\top b`$ is close to 1, before $`b`$ has moved to the minimizer of (12). Together with the warm start this makes SSPO keep the asset it already holds in some periods where another asset has a higher $`R_t`$: 33 of 505 periods on DJIA, 35 of 1,257 on TSE, 54 of 1,274 on SP500. This is the paper's algorithm, not a bug.
- **Few assets.** ADMM converges slowly when $`d`$ is small. On random data it averages about 4,000 iterations per period with two assets and 3,200 with three, and some periods hit the $`10^4`$ cap, against about 700 to 1,200 on the bundled datasets. With a single asset most periods hit the cap, although the weight is always 1. Periods where all assets have the same $`R_t`$, e.g. when all prices are constant, can also run to the cap.
- **Late-listed assets.** With `REPLACE_MISSING`, an asset's price is constant before it is listed, so its $`R_t = 1`$ is the lowest possible and it isn't held under the peak rule unless every asset is at its peak.
- A price that drops to zero makes $`R_t`$ infinite.
