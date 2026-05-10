# Methods — modelling and explainability pipeline

This document summarises the implementation choices made for the
linear (ElasticNet) and gradient-boosted (GBDT / LightGBM)
models and their SHAP-based explainability pipeline.
It is intended as a reference for updating the paper's Methods section.

---

## 1. Cross-validation strategy

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

## 2. Gradient-boosted decision trees (GBDT)

The GBDT model uses **LightGBM** with hyperparameters selected by
Optuna (TPE sampler, 5-fold inner CV, 100 trials).
SHAP values are computed with `TreeExplainer` using the
`tree_path_dependent` perturbation mode, which respects the tree
structure and does not require a background dataset.

---

## 3. ElasticNet linear model

### 3.1 Feature preprocessing (per outer fold, fit on training data only)

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

### 3.2 Hyperparameter selection

Regularisation hyperparameters are selected by **ElasticNetCV** (sklearn
warm-started coordinate-descent path algorithm) using the same 5-fold
grouped splits as the outer loop. The search covers:

- **α** (regularisation strength): 100 values on a log scale spanning
  [10⁻², 10²] (`np.logspace(-2, 2, 100)`).
- **ℓ₁ ratio**: {0.5, 0.7, 0.9, 0.95, 0.99}, interpolating between
  Ridge (ℓ₁ ratio = 0) and Lasso (ℓ₁ ratio = 1).

The warm-started path algorithm is used for hyperparameter selection
because it is orders of magnitude faster than solving each
(α, ℓ₁ ratio, fold) combination independently.

### 3.3 Final model fit

Given the hyperparameters selected in §3.2, the final model for each
outer fold is fitted using sklearn's **`ElasticNet`** (coordinate
descent, `max_iter=100 000`). The high iteration cap ensures convergence
on the ill-conditioned feature subsets that can arise after temporal
blocking removes large contiguous blocks of data.

---

## 4. SHAP explainability

SHAP values are computed on the **full dataset** (all folds combined)
using the fitted model from each outer fold.

- **GBDT**: `TreeExplainer` with `tree_path_dependent` perturbation.
- **ElasticNet**: `LinearExplainer` with an `Independent` masker built
  from a 1 000-sample background set drawn from the training data.
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

## 5. Ablation studies

Four feature sets are evaluated for each model:

| Label | Features included |
|---|---|
| All features | Full feature set |
| No defoliation | All features except defoliation indicators |
| Tree-level only | Tree-level features only (diameter, age, species, …) |
| Plot-level only | Plot-level features only (climate, soil, deposition, …) |
