# Comprehensive Codebase Audit: Report vs. Implementation Validation

**Document:** `REPORT_VALIDATION_AUDIT.md`  
**Target Asset:** `BTC-USD` | **Interval:** Hourly (`1h`)  
**Auditor:** Senior Quantitative Researcher, Machine Learning Engineer & Performance Auditor  
**Audit Date:** 2026-10-19  
**Status:** Audit Completed — **No production code modified pending user review**

---

## 1. Executive Summary

This audit rigorously inspects the actual codebase of the **CrowdWisdomTrading Multi-Horizon Sentiment Price Prediction Agent** against the newly compiled evaluation report (`reports/evaluation_report.md`), the audit report (`reports/audit_and_optimization_report.md`), and institutional quantitative standards.

All 17 automated tests currently pass (`pytest -v tests/` in 1.99s). However, our deep-dive inspection revealed **7 critical methodological, statistical, and architectural discrepancies** between what is documented/claimed and what is actually implemented in the code.

```
+---------------------------------------------------------------------------------------------------------+
|                                        SUMMARY OF AUDIT FINDINGS                                        |
+---+--------------------------------------+---------------------+----------------------------------------+
| # | Audit Dimension                      | Current Status      | Critical Finding / Severity            |
+---+--------------------------------------+---------------------+----------------------------------------+
| 1 | Target Shifts & Timestamp Semantics  | Partially Correct   | target_timestamp points to open, not   |
|   |                                      |                     | close; h=0 has identical start/target  |
+---+--------------------------------------+---------------------+----------------------------------------+
| 2 | Purging & 5-Bar Embargo              | Incomplete          | Purging h steps exists; 5-bar embargo  |
|   |                                      |                     | is completely missing                  |
+---+--------------------------------------+---------------------+----------------------------------------+
| 3 | Frozen Final 30% Holdout             | Missing             | No holdout set exists; metrics are     |
|   |                                      |                     | pooled out-of-fold walk-forward slices |
+---+--------------------------------------+---------------------+----------------------------------------+
| 4 | Sentiment Data Provenance            | Synthetic Only      | Ablation uses 2,881 synthetic posts;   |
|   |                                      |                     | Not clearly warned in Section 3        |
+---+--------------------------------------+---------------------+----------------------------------------+
| 5 | Suspicious Metrics & Overwrites      | Verified Issues     | app.py retrieves all runs without      |
|   |                                      |                     | run_id filter; duplicate timestamps    |
+---+--------------------------------------+---------------------+----------------------------------------+
| 6 | Backtest Mechanics & Fees            | Mathematical Flaw   | Multi-step returns (h=1) compounded    |
|   |                                      |                     | hourly; double-counts overlapping bars |
+---+--------------------------------------+---------------------+----------------------------------------+
| 7 | Statistical Significance Tests       | Missing             | Zero Diebold-Mariano/CW tests; no FDR  |
|   |                                      |                     | or Bonferroni multiple-test correction |
+---+--------------------------------------+---------------------+----------------------------------------+
```

---

## 2. Priority 1: Target Shifts and Hourly Bar Timestamp Semantics

### Source Files & Functions
- `src/targets.py`: `TargetEngineer.create_targets()`, `TargetEngineer.get_horizon_dataset()`
- `src/validate.py`: `WalkForwardValidator.evaluate_model_horizon()` (lines 135–150)
- `src/ingestion/prices.py`: `PriceIngestion.validate_prices()`

### Verified Implementation Mechanics
In `src/targets.py`:
```python
# Cutoff price at start of bar t
if "open" in out.columns:
    out["cutoff_price"] = out["open"]
else:
    out["cutoff_price"] = close.shift(1)

# Horizon targets
out["target_price_h0"] = close
out["target_return_h0"] = (close - p_cut) / p_cut

for h in [1, 2, 3, 4, 5]:
    out[f"target_price_h{h}"] = close.shift(-h)
    out[f"target_return_h{h}"] = (out[f"target_price_h{h}"] - p_cut) / p_cut
```

In `get_horizon_dataset()`:
```python
X_df = df_with_targets[feature_cols].shift(1)
```

### Issues Identified

#### Issue 1.1: `target_timestamp` Points to Bar Open, Creating Misleading Zero-Duration Timestamps
In `src/validate.py` line 145:
```python
target_ts = ts + pd.Timedelta(hours=horizon)
```
- In `yfinance` and cryptocurrency feeds, hourly bars are indexed by their **OPEN** timestamp.
  - A bar with timestamp `2026-05-12 19:00:00` represents trading from `19:00:00` to `19:59:59`.
  - Its `open` is at `19:00:00`. Its `close` is at `19:59:59` / `20:00:00`.
- For horizon $n+0$: `forecast_timestamp = 2026-05-12 19:00:00`, `target_timestamp = 2026-05-12 19:00:00`!
- The target price being predicted is `close(19:00)`, which realizes at `20:00:00`.
- For horizon $n+1$: `forecast_timestamp = 19:00:00`, `target_timestamp = 20:00:00`. But the predicted close is `close(20:00)`, which realizes at `21:00:00`.
- **Impact:** An external reviewer inspecting SQLite `model_predictions` sees identical timestamps for forecast and target at $n+0$, giving the false impression of lookahead leakage or zero-horizon triviality.

#### Issue 1.2: Discrepancy between $P_{open}(t)$ and $P_{close}(t-1)$
In continuous 24/7 crypto markets, $P_{open}(t)$ is theoretically equal to $P_{close}(t-1)$. However, in empirical `yfinance` data:
- `2026-04-22 00:00:00` Close: $76,337.16
- `2026-04-22 01:00:00` Open: $76,370.58 (+$33.42 gap)
Because `p_cutoff` uses `open(t)` while features use `close.shift(1)` (which is `close(t-1)`), the return denominator is not perfectly synchronized with the feature base price.

### Recommended Fix
1. Update `src/validate.py` line 145 to assign `target_ts = ts + pd.Timedelta(hours=horizon + 1)`, correctly reflecting the realization time of the target bar's close.
2. Formally standardize `p_cutoff` as `close.shift(1)` (or explicitly document the $open(t)$ execution assumption).

---

## 3. Priority 2: Five-Bar Embargo vs. Actual Target-Label Intervals

### Source Files & Functions
- `src/validate.py`: `WalkForwardValidator.evaluate_model_horizon()` (lines 92–104)

### Verified Implementation Mechanics
In `src/validate.py`:
```python
purge_steps = horizon

while split_idx < n_samples:
    test_end_idx = min(split_idx + self.test_size, n_samples)
    train_start_idx = 0 if self.expanding else max(0, split_idx - self.min_train_size)
    train_end_idx = max(train_start_idx + 24, split_idx - purge_steps)
```

### Issues Identified

#### Issue 2.1: Purging is Implemented, but Embargo is Completely Absent
- **Purging vs. Embargo Distinction (López de Prado, 2018):**
  - **Purging:** Removes training observations whose target labels overlap with test period timestamps.
  - **Embargo:** An additional buffer period placed immediately before or after test periods to eliminate auto-regressive memory (feature leakage) from rolling indicators.
- In `src/validate.py`:
  - For $h=5$: `purge_steps = 5`. The last training row is `split_idx - 6`. Its target is $close(split\_idx - 6 + 5) = close(split\_idx - 1)$. This correctly purges label overlap.
  - **However, there is zero embargo.**
  - Many features in `src/features.py` have a 24-period lookback window (e.g., `price_volatility_24`, `sma_ratio_24`, `volume_ratio_24`, `sentiment_rolling_mean_24`).
  - The feature vector for training row `split_idx - 6` overlaps with the test sample at `split_idx` across 18 common hours!
- For $h=0$: `purge_steps = 0`, meaning training data immediately borders the test split with zero embargo.

### Recommended Fix
Add an explicit `embargo_bars: int = 5` parameter to `WalkForwardValidator`.
For horizon $h$, the training boundary should be:
$$\text{train\_end\_idx} = \max(\text{train\_start\_idx} + 24, \text{split\_idx} - h - \text{embargo\_bars})$$

---

## 4. Priority 3: Absence of Frozen Final 30% Chronological Holdout

### Source Files & Functions
- `src/validate.py`: `WalkForwardValidator.run_full_validation()`
- `scripts/run_demo.py`: lines 125–150
- `reports/evaluation_report.md`: Section 2

### Verified Implementation Mechanics
- Total available samples: 4,301 contiguous hours (~180 days).
- `min_train_size = 480` (20 days).
- Test step size: `test_size = 168` (7 days).
- In `src/validate.py`, the walk-forward loop iterates across all folds from sample 480 to 4,301.
- All out-of-sample fold predictions are concatenated into a single pool of 3,821 observations.
- All reported metrics in `reports/evaluation_report.md` (Table 2) represent the aggregated walk-forward test folds across the entire dataset.

### Issues Identified

#### Issue 3.1: No Dedicated, Frozen Holdout Dataset Exists
- In quantitative assessment criteria:
  - Phase 1: Model development, hyperparameter tuning, and feature selection on developmental data (e.g., first 70% = 3,010 bars).
  - Phase 2: Final out-of-sample evaluation on a strictly untouched, frozen chronological holdout (e.g., final 30% = 1,291 bars).
- In the current repository, hyperparameters (Ridge $\alpha$, RF tree constraints, HGB L2) were tuned/evaluated on the same walk-forward test folds that are reported as the "final evaluation."
- The report claims: *"Evaluated across 3,821 contiguous out-of-sample hourly test bars (180 days)"*, but fails to clarify that this is a concatenated walk-forward series rather than an untouched holdout set.

### Recommended Fix
1. Explicitly partition the dataset into:
   - **Walk-Forward Development Window (70% = 3,010 bars):** Used for walk-forward CV, hyperparameter tuning, and ablation.
   - **Frozen Chronological Holdout (30% = 1,291 bars):** Quarantined until models are finalized, then evaluated once.
2. Alternatively, if reporting full walk-forward performance, update the documentation to clearly state: *"Metrics represent pooled out-of-fold predictions from walk-forward cross-validation across 23 rolling folds; no single holdout slice was quarantined."*

---

## 5. Priority 4: Sentiment Ablation Uses Synthetic/Sample Data Exclusively

### Source Files & Functions
- `scripts/run_demo.py`: lines 73–85
- `scripts/generate_sample_data.py`: lines 19–47 (`POST_TEMPLATES`)
- `reports/evaluation_report.md`: Section 3 ("Feature Ablation Study")

### Verified Implementation Mechanics
In `scripts/run_demo.py`:
```python
sample_json = cfg.sample_dir / "btc_sample_x_posts.json"
importer = XPostImporter(db_manager=db)
posts = importer.import_from_json(sample_json, source_label="sample_reproducible")
```
- In `scripts/generate_sample_data.py`, `btc_sample_x_posts.json` is generated by sampling from 22 synthetic post templates (8 bullish, 7 neutral, 7 bearish).
- `run_demo.py` performs the entire feature engineering, walk-forward validation, and ablation study using this synthetic file.

### Issues Identified

#### Issue 4.1: Report Section 3 Presents Synthetic Sentiment Ablation without Prominent Warning
In `reports/evaluation_report.md`:
- Section 3 presents Table 3: "Feature Ablation Study: Price-Only vs. Price + Sentiment", showing Random Forest MAE improving from $267.03 to $266.81 (-$0.22) and directional accuracy improving from 50.8% to 51.1%.
- Section 3 does NOT state that the sentiment features were computed from synthetic data.
- While Section 6 ("Free vs Commercial Data Source Audit") notes "Calibrated Reproducible Dataset", presenting ablation metrics in Section 3 without an inline disclaimer violates the constraint:
  > *"Never fabricate data or claim that sample sentiment represents actual market sentiment."*
- **Assessment Risk:** An interviewer or assessor will immediately scrutinize where 180 days of hourly X data came from for free. If they discover it was synthetic while the report discusses genuine sentiment predictive power, credibility is destroyed.

### Recommended Fix
Add a prominent bold alert callout directly inside Section 3 of `reports/evaluation_report.md` and `README.md`:
> **[!IMPORTANT] Data Provenance Disclaimer:** The social sentiment ablation reported below was conducted using the calibrated offline reproducible sample dataset (2,881 synthetic posts), NOT live historical X Corporation firehose data. In an unpaywalled environment, these metrics demonstrate that the pipeline mechanics and ablation code execute correctly; they do NOT prove live social media alpha.

---

## 6. Priority 5: Suspicious Horizon Metrics & Prediction Overwrites

### Source Files & Functions
- `src/database.py`: `get_predictions()`, `save_predictions()`, `save_evaluation_results()`
- `dashboard/app.py`: line 97 (`preds_df = db.get_predictions()`)
- `database/schema.sql`: tables `model_predictions` and `evaluation_results`

### Verified Implementation Mechanics & Database Query Findings

#### Issue 5.1: Duplicate Records and Overwrites in Dashboard
In `src/database.py`:
```python
def get_predictions(self, run_id: Optional[str] = None, model_name: Optional[str] = None, horizon: Optional[int] = None):
    query = "SELECT * FROM model_predictions WHERE 1=1"
    if run_id:
        query += " AND run_id = ?"
        params.append(run_id)
```
In `dashboard/app.py` line 97:
```python
preds_df = db.get_predictions()  # No run_id passed!
```
- Querying SQLite directly revealed 3 separate runs currently residing in `model_predictions`:
  1. `demo_20261019_160843` (old run)
  2. `demo_20261019_163052` (latest main run)
  3. `demo_20261019_163052_price_only` (ablation run)
- For timestamp `2026-05-12 19:00:00`, `db.get_predictions()` returns **3 duplicate rows** for `random_forest` at horizon 1!
- In `dashboard/app.py`, this causes charts to plot multiple conflicting predictions for the same timestamp and inflates metric calculations.
- In `database/schema.sql`, `evaluation_results` has NO unique constraint; every demo run appends 24 rows, causing `db.get_evaluation_results()` to return repeated historical runs.

#### Issue 5.2: Suspicious Metric Proximity at Horizon $n+3$
In `reports/evaluation_report.md`:
- Horizon $n+3$: HistGradientBoosting MAE is **$380.750364**; Ridge MAE is **$380.757416**.
- The difference is less than $0.01 ($0.007).
- **Diagnosis:** Both Ridge (with cross-validated $\alpha$ shrinking toward 1000) and HistGradientBoosting (with L2 regularization = 25.0) predicted near-zero returns at horizon 3 (average absolute predicted return = 0.065%). Because both models predicted $\hat{R} \approx 0$, their price forecasts collapsed to $P_{cutoff}$, causing their MAE to converge to within $0.01 of each other. This is not a software bug, but it indicates extreme model shrinkage where both models act essentially as the baseline.

### Recommended Fix
1. Update `dashboard/app.py` line 97:
   ```python
   latest_run_id = db.get_connection().execute("SELECT run_id FROM evaluation_results ORDER BY id DESC LIMIT 1").fetchone()[0]
   preds_df = db.get_predictions(run_id=latest_run_id)
   eval_df = db.get_evaluation_results(run_id=latest_run_id)
   ```
2. Add a UNIQUE constraint or run-cleanup logic in `database/schema.sql` on `evaluation_results (run_id, model_name, horizon)`.

---

## 7. Priority 6: Backtest Entry/Exit Timing, Fees, Annualization, and Drawdown

### Source Files & Functions
- `src/backtest.py`: `StrategyBacktester.run_backtest()` (lines 97–170)

### Verified Implementation Mechanics
In `src/backtest.py`:
```python
# 1. Trading signals
positions = np.zeros(n, dtype=float)
positions[pred_rets > self.signal_threshold] = 1.0

# 2. Position changes incur transaction costs
pos_shift = np.roll(positions, 1)
pos_shift[0] = 0.0
trades = np.abs(positions - pos_shift)
trade_costs = trades * self.transaction_cost_rate

# 3. Strategy period returns
strat_returns = (positions * actual_rets) - trade_costs

# 4. Cumulative equity curves
equity = self.initial_capital * np.cumprod(1.0 + strat_returns)
benchmark_equity = self.initial_capital * np.cumprod(1.0 + actual_rets)
```

### Issues Identified

#### Issue 6.1: The Overlapping Multi-Period Return Compounding Flaw (CRITICAL)
- The backtest is run on horizon $n+1$ predictions.
- What is `actual_rets` for horizon $n+1$?
  In `src/targets.py`:
  $$\text{actual\_ret}_t^{(1)} = \frac{P_{close}(t+1) - P_{open}(t)}{P_{open}(t)}$$
  This is a **2-hour multi-period return**!
- In `src/backtest.py`:
  - The backtest loop steps **1 hour at a time**.
  - At bar $t$, if `position = 1`, the strategy earns the 2-hour return from $t \to t+1$.
  - At bar $t+1$, if `position = 1`, the strategy earns the 2-hour return from $t+1 \to t+2$.
  - Hour $t+1$ is counted in **both** returns!
- `np.cumprod(1.0 + strat_returns)` compounds 2-hour returns on a 1-hour frequency, resulting in **fictitious leverage and double-counting of returns**!
- If the backtest steps hourly, the trade return earned between bar $t$ and bar $t+1$ MUST be the 1-bar spot return:
  $$r_{t \to t+1} = \frac{P_{close}(t) - P_{open}(t)}{P_{open}(t)} \quad \text{or} \quad \frac{P_{close}(t+1) - P_{close}(t)}{P_{close}(t)}$$
  Using the multi-step target label as the single-step holding return is mathematically invalid.

#### Issue 6.2: Annualization Multiplier Distortion
In `src/backtest.py`:
```python
self.periods_per_year = 8766.0 if frequency == "1h" else 365.25
ann_return = mean_ret * self.periods_per_year * 100.0
ann_vol = std_ret * np.sqrt(self.periods_per_year) * 100.0
sharpe = float(np.mean(excess_returns) / std_ret * np.sqrt(self.periods_per_year))
```
- Because `strat_returns` represents 2-hour holding returns evaluated hourly:
  1. The number of non-overlapping 2-hour periods per year is $8766 / 2 = 4383$. Using $\sqrt{8766}$ inflates the annualization factor by $\sqrt{2} \approx 1.414$.
  2. Overlapping 2-hour returns have strong positive autocorrelation ($\rho_1 \approx 0.50$).
  3. Under Andrew Lo's (2002) classic proof (*The Statistics of Sharpe Ratios*), when returns are autocorrelated, standard Sharpe ratio annualization is biased upward/downward by:
     $$\eta = \sqrt{\frac{q}{1 + 2 \sum_{k=1}^{q-1} (1 - k/q) \rho_k}}$$
     Failing to adjust for overlap autocorrelation invalidates the Sharpe and Sortino statistics.

#### Issue 6.3: Entry/Exit Execution Timing
- At bar $t$, `pred_rets[t]` is evaluated. If `pred_rets[t] > threshold`, `positions[t] = 1.0`.
- The backtest assumes execution occurs at $P_{open}(t)$ without delay.
- However, feature generation and model inference take non-zero compute time at timestamp $t$. Entering at $P_{open}(t)$ assumes zero execution latency.
- In production quant systems, signal generated at bar $t$ executes at $P_{open}(t+1)$ or $P_{close}(t)$.

### Recommended Fix
1. Separate **target forecasting** from **backtest execution**:
   - The model forecasts multi-step return $\hat{R}_{t \to t+1}^{(1)}$.
   - If $\hat{R}_{t \to t+1}^{(1)} > \text{threshold}$, the strategy enters a long position.
   - The return earned between hour $t$ and $t+1$ must be the single-period return $R_{1h}(t)$, NOT the multi-period target return.
2. If holding for the full 2 hours, step the backtest every 2 hours, or track a proper portfolio cash-and-asset state machine with entry price, holding timer, and exit price.

---

## 8. Priority 7: Statistical Tests & Multiple-Comparison Corrections

### Source Files & Functions
- `src/validate.py`
- `src/report.py`
- `reports/evaluation_report.md`

### Verified Implementation Mechanics
- Scanned entire repository for: `diebold`, `mariano`, `wilcoxon`, `clark_west`, `bonferroni`, `holm`, `benjamini`, `p_value`.
- **Result:** Zero statistical hypothesis testing functions or multiple-comparison corrections exist in the codebase.

### Issues Identified

#### Issue 7.1: Zero Statistical Hypothesis Tests Comparing Forecasts to Baseline
- In academic and institutional quantitative finance, showing that Model MAE is $266.81 vs Baseline MAE $265.54 is incomplete without testing whether the loss differential is statistically significant.
- **Diebold-Mariano Test (1995):** The standard test for comparing predictive accuracy between two time-series forecasts:
  $$d_t = L(e_{1,t}) - L(e_{2,t})$$
  $$DM = \frac{\bar{d}}{\sqrt{\hat{V}(\bar{d}) / T}} \sim \mathcal{N}(0, 1)$$
- **Clark-West Test (2007):** Specifically designed for nested models predicting financial returns (where the null is the zero-return persistence martingale).
- Without DM or CW test statistics, the report cannot substantiate whether the small differences in MAE or RMSE are signal or random noise.

#### Issue 7.2: No Multiple-Comparison Correction Across 24 Hypotheses
- The system evaluates **4 models across 6 horizons = 24 simultaneous hypotheses**.
- Under Family-Wise Error Rate (FWER) or False Discovery Rate (FDR):
  - At standard significance level $\alpha = 0.05$, the probability of at least one false positive among 24 tests is:
    $$P(\text{False Positive}) = 1 - (1 - 0.05)^{24} \approx 70.8\%$$
  - Bonferroni corrected significance threshold: $\alpha^* = \frac{0.05}{24} = 0.00208$.
  - Benjamini-Hochberg FDR controls false discoveries across multiple horizons.
- Without these corrections, reporting directional accuracy of 51.5% or 52.9% as an "edge" risks presenting statistical artifacts as findings. For $N = 3,820$:
  $$SE = \sqrt{\frac{0.5 \times 0.5}{3820}} \approx 0.0081 \implies Z = \frac{0.511 - 0.500}{0.0081} = 1.36 \quad (p = 0.087, \text{not significant})$$

### Recommended Fix
1. Implement `diebold_mariano_test(actual, pred_model, pred_baseline, horizon)` in `src/validate.py`.
2. Implement Benjamini-Hochberg FDR and paired Wilcoxon signed-rank tests for loss differentials.
3. Include p-values and significance flags ($* p < 0.05$, $** p < 0.01$) in `reports/evaluation_report.md`.

---

## 9. Comprehensive Traceability Matrix: Issues vs. Files

| Priority | Issue Description | Primary Source File | Functions Affected | Severity |
| :---: | :--- | :--- | :--- | :---: |
| **P1** | `target_timestamp` points to open instead of close; $h=0$ has identical start/target timestamps | `src/validate.py` | `evaluate_model_horizon()` | **High** |
| **P1** | $P_{open}(t)$ vs $P_{close}(t-1)$ tick discrepancy in cutoff price | `src/targets.py` | `create_targets()`, `get_horizon_dataset()` | **Medium** |
| **P2** | Purging exists for $h$ steps, but 5-bar embargo is completely absent | `src/validate.py` | `evaluate_model_horizon()` | **High** |
| **P3** | No frozen final 30% chronological holdout dataset exists | `src/validate.py`, `scripts/run_demo.py` | `run_full_validation()` | **High** |
| **P4** | Sentiment ablation in Section 3 uses synthetic data without prominent inline disclaimer | `reports/evaluation_report.md`, `README.md` | Report Section 3 | **Critical** |
| **P5** | `db.get_predictions()` called without `run_id`, loading duplicate timestamps into dashboard | `dashboard/app.py`, `src/database.py` | `app.py`, `get_predictions()` | **High** |
| **P5** | `evaluation_results` table lacks unique constraint, appending duplicates on rerun | `database/schema.sql`, `src/database.py` | `save_evaluation_results()` | **Medium** |
| **P6** | Multi-period returns ($h=1$) compounded hourly in backtest, double-counting overlapping bars | `src/backtest.py` | `StrategyBacktester.run_backtest()` | **Critical** |
| **P6** | Annualization factor ($\sqrt{8766}$) unadjusted for return overlap and serial autocorrelation | `src/backtest.py` | `StrategyBacktester.run_backtest()` | **High** |
| **P7** | Zero statistical hypothesis tests (Diebold-Mariano / Clark-West) or multiple testing corrections | `src/validate.py`, `src/report.py` | Validation & Reporting engines | **High** |

---

## 10. Next Steps & Recommended Action Plan

Per the user's explicit instruction:
> *"Do not change production code yet. Do not claim a methodology is correct merely because the report says it is. Stop after the audit and show me the findings before making changes."*

Production code remains unchanged. Upon your approval, the recommended sequential remediation plan is:

1. **Step 1 (Backtest & Target Timing Fixes):**
   - Correct the overlapping return compounding in `src/backtest.py` by applying single-bar spot returns to hourly stepping positions.
   - Adjust `target_timestamp` in `src/validate.py` to point to the close of the target bar (`ts + horizon + 1h`).
2. **Step 2 (Purging & 5-Bar Embargo):**
   - Add explicit `embargo_bars = 5` to `WalkForwardValidator` to eliminate auto-regressive rolling feature overlap across folds.
3. **Step 3 (Frozen 30% Holdout Implementation):**
   - Split dataset chronologically: first 70% for walk-forward CV, final 30% strictly quarantined as frozen holdout.
4. **Step 4 (Statistical Tests):**
   - Implement Diebold-Mariano test and Benjamini-Hochberg FDR adjustments in `src/validate.py`.
5. **Step 5 (Data Transparency & Dashboard Cleanliness):**
   - Filter `get_predictions(run_id=latest)` in `dashboard/app.py`.
   - Add bold data-provenance alert directly into Section 3 of `reports/evaluation_report.md`.
6. **Step 6 (Pipeline Rerun & Test Verification):**
   - Re-run test suite and end-to-end demo to generate corrected figures and reports.
