# PPT (Peak Price Tracking)

PPT predicts that each asset's next price will be its highest price over a short window, and moves the portfolio a fixed distance towards the assets whose current price is furthest below that peak. [`ppt.py`](ppt.py) implements it.

Reference: Z.-R. Lai, D.-Q. Dai, C.-X. Ren, K.-K. Huang, [A Peak Price Tracking-Based Learning System for Portfolio Selection](https://ieeexplore.ieee.org/document/7942104), IEEE Transactions on Neural Networks and Learning Systems 29(7), 2018. Authors' code: [laizhr/PPT](https://github.com/laizhr/PPT).

**The paper's text was not checked.** It is paywalled and we found no open copy. Everything below is written from the authors' reference MATLAB code (`PPT_run.m`, `PPT.m`, `simplex_projection_selfnorm2.m`), which this implementation reproduces to about 1e-13. Equation and theorem numbers are left out for that reason, and the paper's own derivation, claims and recommended parameters are not represented here.

## Setup

There are $`m`$ assets with prices $`p_{t,i}`$ and price relatives $`x_{t,i} = p_{t,i} / p_{t-1,i}`$. The portfolio $`b_t`$ held during period $`t`$ lies on the simplex $`\Delta_m = \{b : b_i \ge 0, \sum_i b_i = 1\}`$, and multiplies wealth by $`b_t \cdot x_t`$. The window length is $`w`$ and the step size $`\epsilon`$.

## Prediction

After period $`t`$, PPT predicts the next price relative of each asset as its peak price over the last $`w`$ prices divided by its current price:

```math
\hat{x}_{t+1,i} = \frac{\max_{0 \le k < w} p_{t-k,i}}{p_{t,i}} \ \ge\ 1 .
```

An asset at its window high predicts 1, and an asset that has fallen from it predicts more. For $`t < w`$, counting the first price as $`p_0`$, the prediction is the last price relative $`\hat{x}_{t+1} = x_t`$ instead, as in `PPT.m`. That covers the first $`w - 1`$ updates, including the one at $`t = w - 1`$, where a full window of $`w`$ prices already exists.

## Update

The prediction is centered, so that the step stays on the hyperplane $`\sum_i b_i = 1`$, and the portfolio moves a distance $`\epsilon`$ along it:

```math
c_t = \left(I - \tfrac{1}{m} \mathbf{1}\mathbf{1}^\top\right) \hat{x}_{t+1}, \qquad
\tilde{b} = \begin{cases} b_t + \epsilon \, \dfrac{c_t}{\lVert c_t \rVert} & c_t \ne 0, \\ b_t & c_t = 0, \end{cases} \qquad
b_{t+1} = \operatorname{proj}_{\Delta_m}(\tilde{b}),
```

where $`\operatorname{proj}_{\Delta_m}`$ is the Euclidean projection onto the simplex. $`c_t`$ is the gradient of the predicted return $`b \cdot \hat{x}_{t+1}`$ projected onto that hyperplane, so this is a normalized gradient step with a fixed length $`\epsilon`$. That reading is ours, from the code.

Structurally PPT is [OLMAR](olmar.py) with a different predictor (window peak instead of window mean) and a fixed step length instead of OLMAR's passive-aggressive one, which only moves when the predicted return is below a threshold. [RPRT](rprt.py) is from the same first author and reweights a similar prediction.

## Parameters and limits

| parameter | default | effect |
|---|---|---|
| `window` ($`w`$) | 5 | prices over which the peak is taken, the current one included |
| `eps` ($`\epsilon`$) | 100 | length of the step before projecting |

`eps = 100` is hardcoded in `PPT.m`. The authors' code takes the window as an argument without a default; 5 is the default of OLMAR and RPRT here.

- $`w = 1`$: the peak is the current price, every prediction is 1, $`c_t = 0`$, and PPT holds uniform weights forever: uniform [CRP](crp.py).
- $`\epsilon \to 0`$: the portfolio never moves from uniform weights, again uniform CRP.
- $`\epsilon \to \infty`$: the projection lands on the vertex of the asset with the largest $`\hat{x}_{t+1,i}`$, all in on the asset furthest below its window peak (split between tied assets).

The default $`\epsilon = 100`$ is far larger than the simplex, whose diameter is $`\sqrt{2}`$, so in practice PPT is close to the $`\epsilon \to \infty`$ limit. On the bundled NYSE (O), DJIA, TSE and SP500 data it holds a single asset on 94-96% of days, never more than three, and switches its main holding on 55-58% of days.

## Guarantees

None are stated here. We could not read the paper, so we don't know what it proves, if anything, and the reference code makes no claims. The update has no regret or universal-consistency guarantee that we know of; like OLMAR and RPRT it bets on a price prediction, and its worth is empirical.

## What this means for wealth

PPT is a bet that assets which have just fallen below their recent peak will recover to it, placed almost all in on one asset at a time. On these datasets that bet paid off hugely before costs, as it does for OLMAR and RPRT, and the gains disappear once trading costs are charged.

Final wealth from `algo().run(tools.dataset(name))` with default parameters, without fees and with `result.fee = 0.001` (0.1% per unit traded):

| dataset | days × assets | PPT | OLMAR | RPRT | CRP | BAH | BestSoFar |
|---|---|---|---|---|---|---|---|
| nyse_o | 5651 × 36 | 1.25e18 | 8.22e16 | 1.25e18 | 26.7 | 14.2 | 0.397 |
| nyse_o, fee 0.1% | | 1.94e15 | 4.21e13 | 2.58e14 | 25.0 | 14.2 | 0.261 |
| djia | 507 × 30 | 3.11 | 2.25 | 1.81 | 0.811 | 0.764 | 0.466 |
| djia, fee 0.1% | | 1.75 | 1.17 | 0.880 | 0.805 | 0.764 | 0.430 |
| tse | 1259 × 88 | 266 | 54.9 | 695 | 1.57 | 1.60 | 0.314 |
| tse, fee 0.1% | | 67.1 | 11.1 | 109 | 1.54 | 1.60 | 0.275 |
| sp500 | 1276 × 25 | 11.0 | 15.8 | 18.3 | 1.64 | 1.33 | 1.76 |
| sp500, fee 0.1% | | 2.67 | 3.03 | 2.90 | 1.61 | 1.33 | 1.66 |

PPT turns over about 1.1 of its wealth per day (sum of absolute weight changes), against 0.01-0.02 for CRP. A fee $`f`$ per unit traded therefore costs it about $`1.1 f`$ of wealth per day, roughly a quarter of its wealth a year at $`f = 0.1\%`$. PPT stays ahead of uniform CRP only while the fee is below about 0.6% on NYSE (O), 0.38% on TSE, 0.24% on DJIA and 0.14% on SP500; at 0.5% it ends with 0.17 on DJIA and 0.009 on SP500. Treat the no-fee results as a statement about the data, which comes from well-known mean-reverting datasets, rather than about a strategy you could trade.

## How the code maps to this

| math | code |
|---|---|
| $`p_t`$ | `history.iloc[-1]`; `PRICE_TYPE = "raw"`, so `history` holds prices normalized to start at 1 |
| $`w`$ | `self.window` |
| $`\epsilon`$ | `self.eps` |
| $`\hat{x}_{t+1}`$ | `predict(history.iloc[-window:])` once `len(history) > window` ($`t \ge w`$), else `history.iloc[-1] / history.iloc[-2]` |
| $`c_t`$ | `x_cent` in `update` |
| $`\operatorname{proj}_{\Delta_m}`$ | `tools.simplex_proj` |

`step` is called after period $`t`$, when `history` ends with $`p_t`$, and returns $`b_{t+1}`$, which the framework holds during period $`t+1`$. The first call is after period 1 (`min_history=1`), because the first period has no price relative; until then the weights are uniform. PPT keeps no state between steps: everything it needs is in `history` and `last_b`.

`REPLACE_MISSING = True` forward-fills missing prices, so a missing price means no change and an asset listed late has a constant price, 1 before its listing date. Its prediction is then 1, the lowest possible after the warm-up, so it rarely gets weight; during the warm-up it can, when the other assets have just fallen. Weight on an asset before its listing earns nothing, like cash.

## Limitations

- **Transaction costs.** See above: the strategy is almost all in on one asset that changes most days.
- **No diversification.** With the default step, the projection ends on a vertex or an edge of the simplex, so wealth rides on one asset at a time.
- **Ties.** If every asset predicts the same relative (all at their window high, or constant prices), $`c_t = 0`$ and the portfolio is held unchanged.
- **Reference-code safeguard not reproduced.** `simplex_projection_selfnorm2.m` divides its input by 10 until no entry exceeds $`10^6`$ in absolute value before projecting, which changes the result. It only triggers for $`\epsilon`$ around $`10^6`$ or larger, and this implementation projects without rescaling.
- **First row of the bundled datasets.** The bundled CSVs hold prices whose first row looks like the first day's price relatives in the OLPS `.mat` files the authors' code reads. The framework normalizes that row to 1, so day 1's return is dropped and the warm-up starts a day later. This applies to every algorithm in the package and moves PPT's final wealth by up to about 5% compared with feeding the `.mat` data to the authors' code.
