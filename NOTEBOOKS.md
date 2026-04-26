# Intrinsic Dimensionality of S&P 500 Returns — Notebook Overview

This document explains what the two analysis notebooks in this repository do and how they fit together. It is the single starting point for anyone (teammate, grader, future self) who wants to understand the project without reading every cell of every notebook.

## Project at a glance

The S&P 500 has ~500 stocks, but those stocks are highly correlated — a covariance matrix estimated naively from daily returns is enormous (≈250,000 numbers) and hopelessly noisy. The classical fix is to assume the *true* covariance is low-rank: a small number of underlying factors (the broad market, sector trends, interest rates, growth-vs-value) drive most variance, and the rest is noise.

This project does three things:
1. **Estimates the intrinsic dimensionality** of S&P 500 daily returns using three complementary methods.
2. **Tracks how that dimensionality moves through time**, especially during crisis regimes (Lehman 2008, COVID 2020).
3. **Turns the dimensionality estimate into a usable artifact** — a denoised covariance matrix and a set of factor portfolios — that downstream consumers (portfolio construction, risk modeling) can actually use.

The work is split across two notebooks. Loosely: **v1 is the methodology; v2 is the deliverable.**

| File | Role |
|---|---|
| [`intrinsic_dimensionality.ipynb`](intrinsic_dimensionality.ipynb) | Estimate dimensionality three ways; rolling-window track; robustness CIs |
| [`intrinsic_dimensionality_v2.ipynb`](intrinsic_dimensionality_v2.ipynb) | Denoised covariance + factor portfolios; quantify the value of cleaning |
| [`SP500MarketStructure.py`](SP500MarketStructure.py) | OO library wrapping the same operations (uses yfinance instead of CRSP) |
| [`Driver.py`](Driver.py) | Demo runner for the .py library |

---

## 1. Motivation

### The dimensionality question

If you have $p = 500$ assets and observe $n$ daily returns each, your sample covariance lives in $\mathbb{R}^{p \times p}$. When $n$ is small relative to $p^2$, *most of what you "see" in $\Sigma$ is sampling noise* — the off-diagonal entries are estimated from very little data. Random Matrix Theory (RMT) makes this precise: under the null hypothesis that returns are i.i.d. noise, the sample covariance's eigenvalues follow the Marchenko-Pastur (MP) distribution. Eigenvalues above its upper edge $\lambda_+$ are statistically distinguishable from noise — those are the "real factors."

The number of such real factors is the **intrinsic dimensionality** of returns. The proposal asks three questions:

1. What *is* that dimensionality?
2. Are the factors economically interpretable (sector, style, geography)?
3. Does the dimensionality change across regimes? (Conventional wisdom: it collapses during crises — when everyone panics, every stock just becomes "stock.")

### The portfolio-construction payoff

The reason anyone outside academia cares: building a minimum-variance portfolio requires inverting $\Sigma$. The naive sample $\Sigma$ has a huge condition number (~$10^6$+) precisely because of the noise eigenvalues, and inverting it produces wildly over-fit portfolios with extreme long-short positions. The standard fix is **RMT cleaning** (Laloux-Bouchaud 1999): keep the signal eigenvalues, replace the noise eigenvalues with their mean, and reconstruct. The cleaned $\Sigma$ inverts cleanly and produces sensible portfolios.

The proposal frames this as the deliverable: *"the resulting low-rank matrix can be used as a stable alternative to a high-dimensional covariance matrix when constructing investment portfolios."*

---

## 2. Data

All data lives in [`data_raw/`](data_raw/) as parquet files derived from CRSP (the academic standard for U.S. equity returns):

| File | Contents |
|---|---|
| `sp500_returns_matrix.parquet` | Daily simple returns, trading-days × PERMNO-labeled tickers, 2000-2024 (~6,300 days × 1,300+ tickers ever in the S&P 500) |
| `sp500_constituent_history.parquet` | S&P 500 membership snapshots over time (so we know who was in the index on any given date) |
| `permno_ticker_map.parquet` | PERMNO ↔ ticker ↔ company name mapping from CRSP `msenames` |

### Cleaning recipe (used in both notebooks)

For any analysis window:
1. Pick an as-of date and a date range.
2. Restrict to the **actual S&P 500 members at that as-of date** (uses `constituent_history`, not just "stocks with returns data" — addresses survivorship bias).
3. Drop assets with any missing return in the window. **No imputation.** Strict but defensible.
4. Cast pandas nullable `Float64` to NumPy `float64` for downstream math.

### A note on universe construction

The rolling analyses re-pick the universe at each window's end date — so a 2008 window uses 2008's S&P 500 members, not 2024's. This avoids comparing apples (the modern index) to oranges (the 2008 index). The .py file in this repo does *not* do this — it picks one as-of universe and uses it across all windows, which leaks lookahead into early periods.

---

## 3. The three intrinsic-dimension estimators

The methodology comes from three different traditions; they answer different (but related) questions.

### 3.1 Random Matrix Theory (RMT) — Marchenko-Pastur edge

**Question:** how many eigenvalues of the empirical covariance are statistically distinguishable from pure noise?

Under the null that returns are i.i.d. with variance $\sigma^2$, the eigenvalues of the sample covariance follow the MP distribution, supported on $[\lambda_-, \lambda_+]$ where:

$$\lambda_\pm = \sigma^2 \left(1 \pm \sqrt{q}\right)^2, \qquad q = p/n.$$

Empirical eigenvalues *above* $\lambda_+$ are signals. The RMT dimension is just the count of such eigenvalues.

**The hard part is estimating $\sigma^2$.** The notebook compares three approaches:

| Method | Description | When it fails |
|---|---|---|
| Bottom-half median | $\sigma^2 = \text{median}(\lambda_{p/2:})$ | Biased low when $q \approx 1$ (bulk extends toward zero) |
| **Iterative trim (Laloux)** | Start with $\sigma^2 = \bar\lambda$, drop eigs above implied $\lambda_+$, recompute as mean of remaining bulk, iterate | Robust; canonical choice in both notebooks |
| MP density fit | $\sigma^2$ minimizing L² distance between empirical bulk histogram and MP density | Fragile with small $p$ due to histogram binning |

The notebooks use the iterative trim throughout. The naive median estimator gives RMT = 270 on the 2018-19 window; the iterative method gives **RMT = 149**, which is consistent with Laloux's original published results.

### 3.2 Participation Ratio

**Question:** how concentrated is the variance across the eigenvalue spectrum?

$$D_{PR} = \frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2}.$$

Continuous-valued. If one eigenvalue dominates, $D_{PR} \to 1$. If all eigenvalues are equal, $D_{PR} = p$. Sensitive to the *shape* of the spectrum, not a hard threshold.

On the 2018-19 window: **PR = 10.24** — variance is spread out as if there were ~10 equal-weight modes.

### 3.3 Levina-Bickel — nonlinear NN-MLE

**Question:** what is the geometric (manifold) dimension of the data, allowing for curvature that linear methods miss?

For each sample $x_i$, with $T_k(i)$ = distance to its $k$-th nearest neighbor:

$$\hat{m}_k(x_i) = \left[ \frac{1}{k-1} \sum_{j=1}^{k-1} \log \frac{T_k(i)}{T_j(i)} \right]^{-1}.$$

Final estimate: average over all samples.

**Interpretation choice:** each *trading day* is a sample in $\mathbb{R}^p$ (asset-return space), so this estimates the manifold dimension of daily-return vectors. On the 2018-19 window: **LB = 29.48** with $k = 10$.

### Why three?

The three estimators are sensitive to different things and *should not be expected to agree*. Their disagreement is itself the result:

| Estimator | Type | Captures |
|---|---|---|
| RMT (MP edge) | integer, linear | # of statistically significant linear factors |
| Participation Ratio | continuous, linear | concentration of variance over the full spectrum |
| Levina-Bickel | continuous, nonlinear | geometric / manifold dimension via NN scaling |

The gap LB − PR is roughly the contribution of *nonlinear* curvature (heavy tails, vol clustering, regime mixtures). A widening gap during crises would be an interesting finding.

---

## 4. `intrinsic_dimensionality.ipynb` — v1: the methodology

**Standalone notebook, 25 cells.** Establishes the three estimators on a single window and tracks them through time.

### Section 1 — Data loading & cleaning
Loads the three CRSP parquets, builds three helpers:
- `members_as_of(date)` — binary search on snapshot dates → set of S&P member tickers
- `members_to_columns(tickers)` — vectorized lookup from member set → returns_full columns
- `col_to_company` dict — column label → company name

### Section 2 — Single-window setup (2018-2019)
Picks `AS_OF_DATE = "2019-12-31"` with a 2-year window. Pre-COVID baseline, $q < 1$, no regime contamination.

After cleaning: $n = 503, p = 489, q = 0.972$.

### Section 3 — Method 1: RMT
Compares bottom-half median, iterative trim, and MP density fit estimators for $\sigma^2$. Picks iterative trim as canonical. Plots eigenvalue spectrum and bulk density vs. theoretical MP curve.

**Result:** RMT dim = 149. Top eigenvalue accounts for 30.1% of total variance.

### Section 4 — Method 2: Participation Ratio
One-line computation. Result: **PR = 10.24**.

### Section 5 — Method 3: Levina-Bickel
Vectorized via sklearn's `NearestNeighbors`. Result: **LB = 29.48** (k=10, averaged over 503 daily samples).

### Section 6 — Comparison table
Side-by-side view of the three numbers with their type tags (linear/nonlinear, integer/continuous).

### Section 7 — PC1 loadings (sanity check)
Lists top 10 names by absolute weight on PC1. Result: **all 489 PC1 loadings have the same sign** — textbook market mode. Top names are AMD, NVDA, MU, IPGP, WDC, LRCX, AMAT — high-beta semiconductor cyclicals (PCA conflates the market factor with sector beta amplification).

### Section 8 — Rolling-window comparison
252-day windows stepped quarterly across 2000-2024. **Per-window membership** (uses S&P constituents at each window's end date, not a fixed universe). Computes all three estimators plus `top_eig_share` per window. Plot has Lehman/COVID markers.

### Section 9 — Robustness: 80% subsampling stress test
B = 100 replicates of randomly drawing 80% of the 489 assets. Reports per-estimator mean / std / coefficient of variation. Tells you which estimator is robust to which assets you happen to draw.

### Section 10 — Robustness: moving-block bootstrap CIs
B = 200 replicates with 20-day blocks (preserves volatility clustering). Yields 95% percentile intervals for all three estimators. Plots distribution histograms with point estimate and CI lines.

### Headline v1 numbers (2018-2019 window)

| Estimator | Value | Type |
|---|---|---|
| RMT (iterative σ²) | 149 | integer / linear |
| Participation Ratio | 10.24 | continuous / linear |
| Levina-Bickel (k=10) | 29.48 | continuous / nonlinear |

The 10× spread between PR and LB is the characteristic "the methods don't measure the same thing" finding. Bootstrap CIs (executed in Section 10) put error bars on each.

---

## 5. `intrinsic_dimensionality_v2.ipynb` — v2: the deliverable

**Standalone notebook, 20 cells.** Borrows `rmt_clean_covariance` and `eigenportfolio` from `SP500MarketStructure.py` and uses them to turn dimensionality into a usable artifact: a denoised covariance matrix and a set of factor portfolios.

### Section 1 — Data loading
Same as v1 (mirrored so v2 is standalone).

### Section 2 — Single-window setup
Same 2018-19 window as v1.

### Section 3 — σ² + the three estimators (recomputed)
Same iterative trim from v1, recomputed here so the notebook stands alone. Reproduces RMT=149, PR=10.24, LB=29.48.

### Section 4 — RMT-cleaned covariance

Take the eigendecomposition $\Sigma = V \Lambda V^\top$. Replace every eigenvalue below $\lambda_+$ with the mean of the noise eigenvalues, then reconstruct:

$$\hat{\Sigma}_{\text{clean}} = V \, \text{diag}(\tilde\lambda) \, V^\top, \qquad \tilde\lambda_i = \begin{cases} \lambda_i & \lambda_i > \lambda_+ \\ \bar\lambda_{\text{noise}} & \text{otherwise} \end{cases}$$

Total variance is preserved by construction ($\sum \tilde\lambda_i = \sum \lambda_i$). The signal eigenvectors are intact. The noise spread that destabilizes $\Sigma^{-1}$ is gone.

**Result:**
- 149 of 489 eigenvalues kept as signal
- Trace preserved exactly (1.529194e-01 → 1.529194e-01)
- **Condition number: 4.93×10⁶ → 1.12×10³  (~4400× better)**

### Section 5 — Global Minimum-Variance portfolio comparison

The closed-form GMV portfolio (no constraints, weights sum to 1):

$$w^\star = \frac{\Sigma^{-1} \mathbf{1}}{\mathbf{1}^\top \Sigma^{-1} \mathbf{1}}, \qquad \sigma^2_{w^\star} = \frac{1}{\mathbf{1}^\top \Sigma^{-1} \mathbf{1}}.$$

We construct $w^\star$ from both $\Sigma_{\text{raw}}$ and $\Sigma_{\text{cleaned}}$.

| Metric | GMV(raw) | GMV(cleaned) |
|---|---:|---:|
| Ex-ante variance under own cov | 3.82e-7 | 7.99e-6 |
| Ex-ante variance under *other* cov | 7.54e-4 | 9.10e-6 |
| Gross leverage Σ\|w\| | **68.30** | **6.48** |
| Max single weight | 0.85 | 0.06 |
| Effective # names (1/Σw²) | 0.05 | 7.25 |
| # negative weights | 237 | 222 |

The interpretation:
- **GMV(raw) in-sample variance is lower than GMV(cleaned) in-sample by construction** — but that's the variance of an over-fit portfolio. The cross-cov columns are the honest comparison: GMV(raw) evaluated under $\Sigma_{\text{cleaned}}$ has variance ~80× higher than GMV(cleaned)'s in-sample variance.
- **Gross leverage drops 10.5×.** The raw inverse blows up small noise eigenvalues, producing wildly leveraged long-short bets. Cleaning kills this.
- **Effective # names goes from 0.05 to 7.25.** GMV(raw) puts 85% of its weight in a single name — that's an artifact, not a signal.

### Section 6 — Eigenportfolios

L1-normalize each PC eigenvector so $\sum_i |w_i| = 1$ — gives a tradeable factor portfolio. Show top 5 long and top 5 short positions per component.

| | Top long | Top short | Economic identity |
|---|---|---|---|
| **PC1** (30.1% var) | utilities (~0) | AMD, NVDA, MU, URI, IPGP | Market mode (high-beta semis amplified) |
| **PC2** (4.5%) | WELL, VTR, AMT, O, SBAC (REITs) | APA, XEC, DVN, MRO, NBL (oil & gas) | Rates / sector contrast |
| **PC3** (3.0%) | HBI, CPB, JWN, TAP (defensives, retail) | AMD, NOW, NVDA, MU, ANET (growth tech) | Value vs. growth |
| **PC4** (2.6%) | MRO, DVN, APA, HES, XEC (energy) | AAL, CPRI, UA (consumer cyclicals) | Energy basket |
| **PC5** (2.3%) | UAA, UA, AAL, HBAN, ALK | APA, HES, MRO, DVN | Cyclicals vs. energy (PC4 sign-flipped) |

The clean economic identities are a strong sanity check that PCA is recovering real factors, not noise structure. The PC1 contamination by semiconductor stocks shows that *raw* PCA conflates the market factor with sector-level beta amplification — a Fama-French-style market-beta orthogonalization would push semis out of PC1 and clean up the lower components.

### Section 7 — Cumulative returns of PC1-PC5

Multiply daily returns by L1-normalized weights to get each factor's daily portfolio return; cumulate over the window.

Annualized stats on 2018-19:

| | ann return | ann vol | Sharpe | var share |
|---|---:|---:|---:|---:|
| PC1 | -11.0% | 16.7% | -0.66 | 30.1% |
| PC2 | +12.7% | 7.3% | **+1.75** | 4.5% |
| PC3 | -4.4% | 6.2% | -0.71 | 3.0% |
| PC4 | -2.7% | 8.2% | -0.33 | 2.6% |
| PC5 | -4.7% | 7.8% | -0.61 | 2.3% |

PC1 returns negative because the L1-normalization preserved the all-negative-sign convention of the eigenvector (multiplying by −1 recovers the broad market's positive return). PC2's Sharpe of 1.75 reflects REITs ripping while energy got crushed in 2018-19 — historically accurate.

**These are in-sample stats over 2018-19.** They identify what each factor *was* in that window, not what it will do going forward. They are not strategy backtests.

### Section 8 — Rolling track of cleaning value (2002-2024)

For each rolling 504-day (2-year) window, compute the raw and cleaned covariance, derive GMV portfolios from each, and track three diagnostics:

1. `top_eig_share` — variance share of PC1 (regime indicator from v1)
2. **Condition-number ratio** $\kappa(\Sigma_{\text{raw}}) / \kappa(\Sigma_{\text{cleaned}})$ — how much cleaning regularizes the inverse
3. **Gross-leverage ratio** $\Sigma|w_{\text{raw}}| / \Sigma|w_{\text{cleaned}}|$ — how much cleaning tames noise-driven leverage

86 windows, $q$ stays in [0.935, 0.996] across all of them. Sample of early windows:

| window_end | n_assets | q | rmt_dim | top_eig_share | cond_ratio | gross_ratio |
|---|---:|---:|---:|---:|---:|---:|
| 2002-01-07 | 472 | 0.937 | 139 | 24.5% | 620.6 | 5.4 |
| 2002-04-09 | 474 | 0.940 | 155 | 26.6% | 664.0 | 5.8 |
| 2002-07-09 | 478 | 0.948 | 154 | 27.2% | 649.1 | 5.2 |
| 2002-10-07 | 478 | 0.948 | 162 | 28.4% | 605.2 | 5.1 |
| 2003-01-07 | 481 | 0.954 | 162 | 29.6% | **1393.6** | 5.5 |

Cleaning consistently saves ~5× in leverage; the conditioning improvement runs ~600-1400 in this period and rises into stressed regimes.

### Why a 2-year window in v2 vs. 1-year in v1?

The 1-year (252-day) window with ~480 stocks gives $q \approx 1.9$ — the covariance is rank-deficient, $\Sigma_{\text{raw}}^{-1}$ doesn't exist, and the GMV comparison is undefined. The 2-year window keeps $q < 1$ across the entire 22-year history. The trade-off: crisis spikes are smoother than v1's 1-year rolling chart.

### Numerical-robustness fixes when porting from `SP500MarketStructure.py`

When porting `rmt_clean_covariance` and using it at scale, two issues surfaced:

1. **`iter_sigma2` collapsed to ~0** when the spectrum contained numerical-zero eigenvalues from rank deficiency. Fix: filter to strictly positive eigenvalues before trimming.
2. **Cleaned cov could go indefinite** if the noise mean was contaminated by tiny negative numerical noise. Fix: estimate noise mean from positive eigenvalues only; fall back to the smallest signal eig if no positive noise exists.

Both fixes are in v2 and are documented inline.

---

## 6. Cross-cutting findings

Reading both notebooks together, the substantive results are:

### 6.1 The three estimators disagree by 10× — and that's the result
RMT=149, PR=10, LB=30 on the 2018-19 window. They're answering different questions; reconciliation isn't the goal. The story is: ~10 effectively-equal modes carry the variance, ~30 nonlinear directions describe the geometric manifold, and ~149 individual modes can be statistically resolved from noise.

### 6.2 PC1 is unambiguously the market mode
All 489 PC1 loadings have the same sign. PC1 explains 30.1% of variance. Top loadings are high-beta semis, not the broadest names — confirming that raw PCA captures market direction *modulated by individual-stock beta to it*.

### 6.3 PC2-PC5 have clean economic identities
REITs vs. oil & gas (rates), defensives vs. growth tech (value/growth), energy basket, cyclicals vs. energy. These are the factors quants actually trade.

### 6.4 Concentration is regime-sensitive
Top eigenvalue's variance share rises from 23% to 32% across 2001-2002 (dot-com tail / 9/11). The expected behavior: dimensionality collapses during crises.

### 6.5 RMT cleaning is essential, not optional, for portfolio construction
The raw GMV portfolio on 2018-19 takes 68× leverage and concentrates 85% of weight in a single stock. The cleaned GMV takes 6× leverage and has 7 effective names. The cleaning improves matrix conditioning by ~4400×.

### 6.6 Cleaning's value rises in stressed periods
The condition-number ratio (raw/cleaned) goes from ~600 in calm 2002 to ~1400 by Jan 2003. Gross-leverage ratio holds fairly steady at ~5×.

---

## 7. Known issues / things to fix

1. **Ticker-mapping bug in v2 PC4/PC5.** `MS_82621` shows up labeled "Milestone Scientific" with large weights — should be Morgan Stanley. There's a `permno_ticker_map` lookup collision, probably in the `groupby("permno").last()` step. Doesn't change the broad findings (the loading magnitudes themselves are correct, just the company name lookup is wrong) but should be fixed before final writeup.
2. **The Sharpe ratios on the eigenportfolios are in-sample.** They identify what factors *were* in 2018-19, not what they'll do going forward. Treat as economic identification, not strategy claims.
3. **v1's rolling RMT cell uses iter_sigma2 but the notebook's cached outputs may show old (median-σ²) numbers.** Re-execute v1 if you want the up-to-date rolling RMT line.

---

## 8. Glossary

| Term | Meaning |
|---|---|
| **PERMNO** | CRSP's permanent security identifier — survives ticker changes, mergers, etc. |
| **q = p/n** | Aspect ratio: assets divided by trading days. RMT theory is sharpest when $q < 1$. |
| **Marchenko-Pastur (MP) distribution** | Limiting eigenvalue density of a sample covariance built from i.i.d. noise |
| **MP edge $\lambda_+$** | Upper bound of the MP support: $\sigma^2(1+\sqrt{q})^2$. Eigenvalues above this are signals. |
| **σ²** | Average per-asset noise variance — the parameter that sets the MP scale. |
| **Iterative trim (Laloux)** | Robust σ² estimator: drop signal eigs, recompute σ² as mean of remainder, iterate. |
| **PR (Participation Ratio)** | $(\Sigma\lambda)^2 / \Sigma\lambda^2$ — continuous concentration measure. |
| **LB (Levina-Bickel)** | Nearest-neighbor MLE of manifold dimension. |
| **PC$k$** | $k$-th principal component (eigenvector of cov sorted by eigenvalue). |
| **Eigenportfolio** | A PC eigenvector treated as a normalized portfolio weight vector. |
| **GMV** | Global minimum-variance portfolio: $w^\star = \Sigma^{-1}\mathbf{1} / (\mathbf{1}^\top\Sigma^{-1}\mathbf{1})$. |
| **Gross leverage** | $\sum_i \|w_i\|$ — total notional exposure including shorts. |
| **Condition number** | $\lambda_{\max} / \lambda_{\min}$ — sensitivity of $\Sigma^{-1}$ to small perturbations. |
| **Block bootstrap** | Resample contiguous blocks of observations to preserve short-range autocorrelation. |
| **Survivorship bias** | Analyzing only currently-listed stocks — produces upward-biased return estimates. |

---

## 9. Reading order

If you have 30 minutes:
1. This document (~10 min)
2. v1 sections 1, 2, 6, 8 (the data setup, the three estimators side-by-side, the rolling chart) (~10 min)
3. v2 sections 5, 6, 8 (the GMV comparison, the eigenportfolio identities, the rolling cleaning-value track) (~10 min)

If you have 5 minutes: read sections 6 ("Cross-cutting findings") and 7 ("Known issues") of this document.
