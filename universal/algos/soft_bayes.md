# Soft-Bayes

Soft-Bayes is the Prod algorithm applied to the linearised log-loss. [`soft_bayes.py`](soft_bayes.py) applies it to portfolio selection, with each asset as an expert and a constant rebalanced portfolio (CRP) as the comparator. Each period it moves the weights a fraction $`\eta`$ of the way towards where last period's returns let them drift. Its regret against the best CRP is $`O(\sqrt{TN \ln N})`$, and unlike EG's, that bound doesn't depend on how large the price relatives get. It costs $`O(N)`$ per period.

Reference: L. Orseau, T. Lattimore, S. Legg, [Soft-Bayes: Prod for Mixtures of Experts with Log-Loss](https://arxiv.org/abs/1901.02230), ALT 2017. Equation and theorem numbers below refer to that paper.

## Setup

There are $`N`$ assets and periods $`t = 1, \dots, T`$. The paper's expert predictions $`p^i_t`$ become price relatives, and the learner's prediction $`M_t`$ becomes the portfolio's return:

```math
p^i_t = x_{t,i} = \frac{S_{t,i}}{S_{t-1,i}}, \qquad M_t = \sum_{i=1}^{N} w^i_t p^i_t = w_t \cdot x_t, \qquad M_{1:T} = \prod_{t=1}^{T} M_t .
```

So $`M_{1:T}`$ is the final wealth of the algorithm, rebalanced every period. A comparator $`a`$ in the simplex is a CRP, its prediction $`A_t = a \cdot x_t`$ is its return, and $`A_{1:T}`$ is its final wealth. The regret (eq. 1) is the log-wealth ratio

```math
\mathcal{R}_T(a) = \ln \frac{A_{1:T}}{M_{1:T}} = \ln W^{\text{CRP}(a)}_T - \ln W^{\text{SoftBayes}}_T .
```

The paper's $`p^i_t`$ are probabilities, but the algorithm and its proofs only use the ratios $`p^i_t / M_t`$ and $`A_t / M_t`$. Both are unchanged when every relative in a period is scaled by the same factor, so the guarantees hold for price relatives as they are.

## The update

Start from uniform weights $`w_1 = 1/N`$. After period $`t`$ (eq. 7):

```math
w^i_{t+1} = w^i_t \left(1 - \eta + \eta \frac{p^i_t}{M_t}\right) = (1 - \eta) w^i_t + \eta \frac{w^i_t p^i_t}{M_t}, \qquad \eta \in (0, 1).
```

The second form says it all: the new weights mix the old weights with the weights after one period of drift, $`w^i_t p^i_t / M_t`$. The weights stay normalised without dividing by a sum. EG uses the same signal, but exponentiated (eq. 8):

```math
\text{EG:} \quad w^i_{t+1} \propto w^i_t \exp\left(\eta \frac{p^i_t}{M_t}\right).
```

When an asset with a tiny weight suddenly jumps, $`p^i_t / M_t`$ is huge and EG moves almost all the wealth into it. Soft-Bayes can't, because $`M_t \ge w^i_t p^i_t`$ (dominance) caps every step:

```math
(1 - \eta) w^i_t \ \le\ w^i_{t+1} \ \le\ (1 - \eta) w^i_t + \eta .
```

No asset gains more than $`\eta`$ of weight or loses more than a fraction $`\eta`$ of it in one period, however extreme the returns.

## What $`\eta`$ does

| $`\eta`$ | weights | strategy |
|---|---|---|
| $`\eta = 0`$ | $`w_t = 1/N`$ | uniform CRP ([CRP](crp.py)) |
| $`\eta = 1`$ | $`w^i_{t+1} = w^i_t p^i_t / M_t \propto S_{t,i} / S_{0,i}`$ | the Bayesian posterior over assets, i.e. buy and hold ([BAH](bah.py)) |

Hence the name: $`\eta = 1`$ is Bayes, and a smaller $`\eta`$ slows Bayes down so that it doesn't commit to one asset too fast. The theorems assume $`\eta \in (0, 1)`$. The code also accepts the two endpoints.

## Guarantees

**Against a single asset** (Theorem 1), for any $`\eta \in (0, 1]`$:

```math
\mathcal{R}_T(e_i) = \ln \frac{p^i_{1:T}}{M_{1:T}} \le \frac{1}{\eta} \ln \frac{1}{w^i_1} = \frac{\ln N}{\eta} .
```

**Against any CRP** (Theorem 3). Let $`\mathcal{M}^* = \{i : \exists t,\ p^i_t = \max_j p^j_t\}`$ be the assets that had the highest return in at least one period, and $`m = |\mathcal{M}^*| \le N`$. Writing $`\bar\eta = \eta / (1 - \eta)`$, for any $`\eta \in (0, 1)`$ and any $`a`$:

```math
\mathcal{R}_T(a) \le \frac{1}{\bar\eta} \ln N + \bar\eta m T + m \ln \frac{N}{m} + \ln N .
```

With $`\bar\eta = \sqrt{\ln N / (T m)}`$ it becomes

```math
\mathcal{R}_T(a) \le 2\sqrt{T m \ln N} + m \ln \frac{N}{m} + \ln N \ \le\ 2\sqrt{T N \ln N} + \ln N .
```

The bound depends only on $`N`$, $`m`$ and $`T`$. EG's bound (eq. 2) is $`C \sqrt{(T/2) \ln N}`$ with $`C = \max_{t,i} p^i_t / M_t`$, which can be arbitrarily large, and Theorem 2 builds a sequence on which EG's regret is nearly linear in $`T`$ for every learning rate.

**Anytime** (Section 6, Theorem 10). The tuning above needs $`T`$ and $`m`$ in advance. With a decreasing rate

```math
\eta_t = \sqrt{\frac{\ln N}{2 N t}}
```

and the online correction of eq. (14), which mixes the prior back in

```math
w^i_{t+1} = w^i_t \left(1 - \eta_t + \eta_t \frac{p^i_t}{M_t}\right) \frac{\eta_{t+1}}{\eta_t} + \left(1 - \frac{\eta_{t+1}}{\eta_t}\right) w^i_1 ,
```

the regret against any CRP is, without knowing $`T`$,

```math
\mathcal{R}_T(a) \le 2\sqrt{2 (T + 1) N \ln N} + \left(\tfrac{1}{2} N + \ln N\right) \ln (T + 1) + \ln N .
```

The correction keeps every weight above $`w^i_1 (1 - \eta_{t+1}/\eta_t)`$ (eq. 16), so an asset that has done badly for a long time can still recover. Without it, the weight of such an asset shrinks like $`\prod_{s \le t} (1 - \eta_s)`$, which the paper shows is too fast for simply plugging a decreasing $`\eta_t`$ into eq. (7). For this schedule $`\eta_{t+1}/\eta_t = \sqrt{t/(t+1)}`$, which is also the cap that Remark 8 recommends, so the remark changes nothing here.

## What this means for wealth

Since the regret is the log-wealth ratio, the bounds say directly that Soft-Bayes' final wealth is at least $`e^{-\mathcal{R}}`$ times that of the best CRP in hindsight (BCRP), with no Jensen gap in between. The bound grows like $`\sqrt{T}`$, so the gap in growth rate per period vanishes. But it is a worst-case bound: on the bundled datasets $`2\sqrt{TN\ln N}`$ is in the hundreds or thousands of nats, far more than the actual gap between BCRP and uniform CRP. The theory-optimal $`\eta`$ is tiny (0.004 to 0.015 there), and the weights never move more than 0.001 away from uniform.

So in practice Soft-Bayes is uniform CRP with a guarantee. On the bundled datasets it finishes within about 0.2% of uniform CRP's wealth:

| dataset | periods, assets | BAH | CRP | BestSoFar | EG | SoftBayes (anytime) | SoftBayes (theory $`\eta`$) | BCRP |
|---|---|---|---|---|---|---|---|---|
| nyse_o | 5651, 36 | 14.21 | 26.68 | 0.397 | 26.70 | 26.67 | 26.68 ($`\eta`$ = 0.0042) | 248.5 |
| djia | 507, 30 | 0.764 | 0.811 | 0.466 | 0.808 | 0.810 | 0.810 ($`\eta`$ = 0.0147) | 1.252 |
| tse | 1259, 88 | 1.596 | 1.570 | 0.314 | 1.569 | 1.570 | 1.570 ($`\eta`$ = 0.0063) | 6.599 |
| sp500 | 1276, 25 | 1.333 | 1.639 | 1.755 | 1.624 | 1.637 | 1.636 ($`\eta`$ = 0.0099) | 4.050 |

Final wealth without fees. The other algorithms use their default parameters (EG $`\eta = 0.05`$). "Theory $`\eta`$" is Theorem 3's with $`m = N`$ and the dataset's own $`T`$. Where it matters is adversarial data. With a leader that flips every period by a factor of 4, buy and hold's regret against $`(\tfrac12, \tfrac12)`$ grows by $`\ln 1.25`$ per period, while Soft-Bayes' regret stays below Theorem 3's $`O(\sqrt{T})`$ bound (see the tests).

## How the code maps to this

| math | code |
|---|---|
| $`p^i_t = x_{t,i}`$ | `x`; `PRICE_TYPE = "ratio"` (the default), so `step` gets price relatives |
| $`w_t`$ | `last_b` in, the returned array out |
| $`M_t`$ | `b @ x` |
| $`\eta`$ | `self.eta` |
| $`\eta_t`$, anytime | `np.sqrt(np.log(n_assets) / (2 * n_assets * self.t))` when `eta=None` |
| $`\eta_{t+1}/\eta_t`$ | `beta = np.sqrt(self.t / (self.t + 1))` |
| $`w_1`$ | `self.w1`, from `init_weights` (uniform unless a subclass overrides it) |
| $`t`$ | `self.t`, the number of `step` calls |

`step(x)` is called after period $`t`$ with that period's relatives $`x_t`$ and the weights $`w_t`$ held during it, and returns $`w_{t+1}`$, which the framework holds during period $`t + 1`$. The framework's first row of relatives is all ones. With a fixed $`\eta`$ that step leaves the weights unchanged. In the anytime variant it counts as round 1 of the schedule, which costs nothing, since its regret is zero.

## Limitations

- An asset without a price yet (listed late) is treated as having a constant price (`REPLACE_MISSING = True`): its share of the weights sits in cash until it lists, and the guarantees apply to that sequence. Delisted assets are treated the same way. The paper assumes every expert predicts in every period.
- Only Theorem 10's schedule is implemented. The variant that tracks the set of leaders online (Section 6.1) and the self-confident learning rates of Theorems 5 and 6 are not.
- The prior is uniform unless a subclass overrides `init_weights`. Theorem 1 holds for any prior, with $`\ln(1 / w_1^i)`$ in place of $`\ln N`$; the other bounds above assume the uniform prior.
- The theory-optimal and anytime learning rates are so small on real data that the algorithm is hard to tell from uniform CRP. Larger values of $`\eta`$ are allowed, but Theorem 3's bound then grows linearly in $`T`$.
