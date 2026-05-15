# Methods — modelling and explainability pipeline

This document summarises the implementation choices made for the
linear (ElasticNet, LMM) and gradient-boosted (GBDT / LightGBM)
models and their SHAP-based explainability pipeline.
It is intended as a reference for updating the paper's Methods section.

---

## 1. Target variable

The response is the **log relative growth rate** (log-RGR) per year, stored as
`growth_rate_rel` in the data.

Before modelling, a **probability integral transform (PIT)** is applied to
stabilise variance and render the marginal target distribution approximately
uniform:

1. **Shift**: y′ = log-RGR + 1 (ensures y′ > 0 across all observed values).
2. **Fit**: a log-normal distribution is fitted to y′ on the full per-species
   dataset (`scipy.stats.lognorm.fit`).
3. **Transform**: ỹ = F_lognorm(y′) ∈ (0, 1).

The PIT gives linear models a better-conditioned regression problem.
SHAP values and model coefficients are expressed in PIT-quantile units;
the notebook provides utilities to map predictions back to the original
log-RGR scale.

The LMM additionally z-scores ỹ on the training fold before fitting (§5.2).

---

## 2. Cross-validation strategy

All models are evaluated with **5-fold grouped cross-validation**.
The grouping column is `tree_id`, so all observations from a given tree
are always assigned to the same fold, preventing data leakage from
repeated measurements.

An additional **temporal blocking** option is available: folds are
constructed so that each test set corresponds to a contiguous time slice
that is always *later* than the training observations in that fold.
This removes temporal autocorrelation from the performance estimates.
All results reported in the paper use temporal blocking.

---

## 3. Gradient-boosted decision trees (GBDT)

The GBDT model uses **LightGBM** with hyperparameters selected by
Optuna (TPE sampler, 5-fold inner CV, 100 trials).
SHAP values are computed with `TreeExplainer` using the
`tree_path_dependent` perturbation mode, which respects the tree
structure and does not require a background dataset.

---

## 4. ElasticNet linear model

### 4.1 Feature preprocessing (per outer fold, fit on training data only)

The following steps are applied in order, each fitted exclusively on the
training portion of the outer fold and then applied to both train and
test:

1. **Missingness filter** — features with more than 50 % missing values
   in the training fold are dropped before any imputation.
   This step is necessary because a feature that is, say, 80 % missing
   still has genuine variance in its observed 20 % of rows and therefore
   passes variance-based filters. After mean imputation it becomes a
   fold-specific constant; under temporal cross-validation the test fold
   can have a different missingness pattern or different actual values,
   causing large distribution shift and unstable out-of-sample
   predictions. Soil-solution chemistry features (`ss_*`) are the
   primary affected group for spruce in the no-defoliation ablation.

2. **Median imputation** — remaining missing values are filled with the
   column median of the training fold (`SimpleImputer(strategy="median")`).

3. **Near-zero variance filter** — features whose variance falls below
   1 × 10⁻⁴ after imputation are dropped
   (`VarianceThreshold(threshold=1e-4)`).

4. **Robust scaling** — features are centred by median and scaled by
   interquartile range (`RobustScaler`), which limits the influence of
   outliers on the regularisation path.

### 4.2 Hyperparameter selection

Regularisation hyperparameters are selected by **ElasticNetCV** (sklearn
warm-started coordinate-descent path algorithm) using the same 5-fold
grouped splits as the outer loop. The search covers:

- **ℓ₁ ratio**: {0.1, 0.5, 0.7, 0.9, 0.95, 0.99}, interpolating between
  Ridge (ℓ₁ ratio = 0) and Lasso (ℓ₁ ratio = 1).
- **α** (regularisation strength): 100 log-spaced values on a data-driven
  interval [α_min, α_max]:
  - **α_max** = max|**X**ᵀ**y**| / n — the value at which all coefficients
    vanish (start of the regularisation path), following the standard
    sklearn convention.
  - **α_min** is a *condition-number floor*: the smallest α for which the
    regularised Gram matrix **X**ᵀ**X** + nα(1 − ℓ₁)·**I** has condition
    number ≤ 10⁴. Rearranging the threshold equation gives

    α_floor = (λ_max − κ · λ_min) / (n · (1 − ℓ₁) · (κ − 1))

    where λ_max and λ_min are the extreme eigenvalues of **X**ᵀ**X** and
    κ = 10⁴. The floor is evaluated at the worst-case ℓ₁ ratio (0.99),
    which minimises the ridge term (1 − ℓ₁) and therefore requires the
    highest α to satisfy the constraint. If **X**ᵀ**X** is already
    well-conditioned (λ_max ≤ κ · λ_min), α_floor = 0 and a hard lower
    bound of α_max × 10⁻⁶ is used.

The warm-started path algorithm is used for hyperparameter selection
because it is orders of magnitude faster than solving each
(α, ℓ₁ ratio, fold) combination independently.

### 4.3 Final model fit

Given the hyperparameters selected in §4.2, the final model for each
outer fold is fitted using sklearn's **`ElasticNet`** (coordinate
descent, `max_iter=10 000`). The high iteration cap ensures convergence
on the ill-conditioned feature subsets that can arise after temporal
blocking removes large contiguous blocks of data.

---

## 5. Mixed-effects linear model (LMM)

### 5.1 Model structure

The LMM is a random-intercept model of the form

```text
y_ij = Xβ + u_j + ε_ij,   u_j ~ N(0, σ²_u),   ε_ij ~ N(0, σ²_e)
```

where *i* indexes an observation, *j* indexes its plot, **X** is the
same preprocessed feature matrix as in the ElasticNet (§4.1), **β** is
the fixed-effects coefficient vector, *u_j* is the plot-specific random
intercept, and ε_ij is the residual.

### 5.2 Target standardisation

Before fitting, the PIT-transformed target ỹ is standardised to zero
mean and unit variance on the training fold.  Predictions are rescaled
back to the PIT-quantile space after fitting.  Standardisation is
necessary to keep the optimiser well-conditioned given the mix of highly
regularised and near-zero variance features that can arise under
temporal blocking.

### 5.3 Feature preprocessing

Identical to §4.1 (missingness filter → median imputation → near-zero
variance filter → RobustScaler), fitted on the training fold only.

### 5.4 Estimation

Parameters are estimated by **restricted maximum likelihood (REML)**,
which gives unbiased variance-component estimates and therefore more
accurate BLUPs.
The implementation uses `statsmodels.regression.mixed_linear_model.MixedLM`.
Optimisers are tried in order — L-BFGS, BFGS, Nelder-Mead — using the
first that achieves convergence.  Non-convergence is logged as a warning
and flagged in the saved hyperparameter artefacts (`converged=False`).

### 5.5 Prediction and BLUP adjustment

Two prediction modes are used depending on context:

**Fixed-effects only** (used for SHAP attribution and `y_pred` storage):

```text
ŷ = Xβ̂
```

This ensures that `LinearExplainer` attribution reconstructs predictions
exactly — the random intercept is not a feature and has no SHAP value.

**BLUP-adjusted** (used for R² / RMSE under tree-wise CV):

```text
ŷ_ij = Xβ̂ + û_j
```

where û_j is the Best Linear Unbiased Predictor of the random intercept
for plot j, taken from statsmodels' `random_effects` after fitting.
Under tree-wise cross-validation the same plot can appear in both
training and test folds (different trees), so the BLUP provides a valid
out-of-sample estimate of the plot effect and yields fairer R² / RMSE
numbers.  Observations whose plot was not seen during training receive
û_j = 0 (population-level fallback).  Under plot-wise CV all test plots
are unseen, so both modes coincide.

### 5.6 Variance components

After fitting, the following quantities are extracted and stored in
`cache/hyperparams-{ablation}-lmm-{group_col}.parquet`:

| Symbol | statsmodels attribute | Description |
|---|---|---|
| σ²_u (`var_random`) | `result.cov_re.iloc[0, 0]` | Between-plot variance |
| σ²_e (`var_resid`) | `result.scale` | Within-plot (residual) variance |
| ICC | σ²_u / (σ²_u + σ²_e) | Intraclass correlation coefficient |

---

## 6. SHAP explainability

SHAP values are computed on the **full dataset** (all folds combined)
using the fitted model from each outer fold.

- **GBDT**: `TreeExplainer` with `tree_path_dependent` perturbation.
- **ElasticNet / LMM**: `LinearExplainer` with an `Independent` masker
  built from a 1 000-sample background set drawn from the training data.
  For the LMM, SHAP is computed on the fixed-effects coefficients only
  (rescaled to the original target space via `_FixedEffectLinear`); the
  random intercept is not included because it is set to zero at
  prediction time for unseen plots.
  Because the preprocessing pipeline changes the feature space
  (missingness filter + variance filter), the SHAP computation is
  performed in the reduced (preprocessed) feature space and the
  resulting values are zero-padded back to the original feature
  dimensionality before storage. Features dropped by the preprocessing
  pipeline receive a SHAP value of zero, which is correct: they
  contribute nothing to model predictions.

Feature importance is reported as the **mean absolute SHAP value**
across all observations and folds, optionally weighted by fold size.

---

## 7. Ablation studies

Four feature sets are evaluated for each model:

| Label | Features included |
|---|---|
| All features | Full feature set |
| No defoliation | All features except defoliation indicators |
| Tree-level only | Tree-level features only (diameter, age, species, …) |
| Plot-level only | Plot-level features only (climate, soil, deposition, …) |
