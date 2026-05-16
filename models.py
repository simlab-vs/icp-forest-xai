# Train models
from __future__ import annotations


from config import Ablation, Species
from lightgbm import LGBMRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.feature_selection import VarianceThreshold

import sklearn
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import mean_squared_error, make_scorer, root_mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from shap import TreeExplainer, Explanation, LinearExplainer, Explainer
from shap.maskers import Independent as IndependentMasker
import joblib
import optuna
from optuna.trial import Trial

import sys
import contextlib
import logging
import os

import numpy as np
import polars as pl
from scipy.stats import lognorm

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence, cast, overload

from data import prepare_data, load_data
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*force_all_finite.*",
    category=FutureWarning,
    module="sklearn",
)
# ConvergenceWarning from the low-alpha tail of the ElasticNet path is expected:
# the path algorithm evaluates all grid points but CV never selects the unconverged ones.
# catch_warnings() is thread-local in Python 3.12+, so this must be a module-level
# filter to be visible to joblib worker threads.
warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning,
    module=r"sklearn\.linear_model\._coordinate_descent",
)

Split = Literal["train", "test", "all"]
ModelType = Literal["gbdt", "elasticnet", "lmm"]
MatrixLike = np.ndarray | pl.DataFrame
VectorLike = np.ndarray | pl.Series

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

RANDOM_STATE = 42
ALL_SPECIES: list[Species] = ["spruce", "pine", "beech", "oak"]


# This is a hack to suppress stderr output from LightGBM
# It is used to avoid cluttering the output with LightGBM's verbose messages.
@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


@overload
def to_numpy(data: MatrixLike | VectorLike) -> np.ndarray: ...
@overload
def to_numpy(data: None) -> None: ...


def to_numpy(data: MatrixLike | VectorLike | None) -> np.ndarray | None:
    """Convert data to a NumPy array if it is a Polars DataFrame or Series."""
    if data is None:
        return None
    if isinstance(data, pl.DataFrame):
        return data.cast(pl.Float64).to_numpy()
    elif isinstance(data, pl.Series):
        return data.to_numpy()
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise TypeError(
            f"Unsupported data type: {type(data)}. Expected DataFrame or Series."
        )


def to_pandas(data: MatrixLike) -> Any:
    """Convert a matrix to a pandas DataFrame, preserving column names.

    Used for LightGBM fit/predict so that feature_names_in_ stays consistent
    and sklearn does not emit 'X does not have valid feature names' warnings.
    LightGBM 4.x auto-assigns Column_0…N when fitted with numpy, then warns
    on every numpy predict because the names don't match.
    """
    import pandas as pd

    if isinstance(data, pl.DataFrame):
        return data.cast(pl.Float64).to_pandas()
    elif isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, np.ndarray):
        return pd.DataFrame(data)
    else:
        raise TypeError(
            f"Unsupported data type: {type(data)}. Expected DataFrame or ndarray."
        )


def r2_score(
    y: VectorLike,
    y_pred: VectorLike,
    *,
    y_ref: VectorLike | None = None,
) -> float:
    """Compute the R2 score based on a given out-of-sample target vector and loss function.

    Parameters
    ----------
    y
        True target values.
    y_pred
        Predicted target values.
    y_ref
        In-sample target values (if not provided, y is used).

    Returns
    -------
    The R2 score.
    """
    y = to_numpy(y)
    y_pred = to_numpy(y_pred)
    y_ref = to_numpy(y_ref)

    # Reference target values used to compute the baselien predictions
    y_ref = y if y_ref is None else y_ref
    y_base = np.full_like(y, np.mean(y_ref))

    return cast(
        float,
        1 - mean_squared_error(y, y_pred) / mean_squared_error(y, y_base),
    )


class EstimatorProtocol(Protocol):
    """Protocol for a regressor that can be used in cross-validation."""

    def fit(self, X: MatrixLike, y: VectorLike, **kwargs: Any) -> EstimatorProtocol:
        """Fit the regressor to the training data."""
        ...

    def predict(self, X: MatrixLike) -> VectorLike:
        """Predict using the fitted regressor."""
        ...

    def get_params(self, deep: bool = True) -> dict[str, Any]: ...

    def set_params(self, **params: Any) -> EstimatorProtocol:
        """Set the parameters of the regressor."""
        ...

    def score(self, X: MatrixLike, y_true: VectorLike) -> float:
        """Compute the score of the regressor on the given data.

        Parameters
        ----------
        X
            Features to predict on.
        y_true
            True target values to compute the score against.

        Returns
        -------
        The R2 score of the regressor on the given data."""
        return r2_score(y_true, self.predict(X))

    def rmse(self, X: MatrixLike, y_true: VectorLike) -> float:
        """Compute the score of the regressor on the given data.

        Parameters
        ----------
        X
            Features to predict on.
        y_true
            True target values to compute the score against.

        Returns
        -------
        The R2 score of the regressor on the given data."""
        return root_mean_squared_error(y_true, self.predict(X))


class LGBMEstimator(EstimatorProtocol):
    """LightGBM regressor."""

    def __init__(
        self,
        *,
        species: Species,
        group_by: str,
        cv: int = 5,
        n_jobs: int = -1,
        random_state: int | None = RANDOM_STATE,
        verbosity: int = -1,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("force_row_wise", None)  # Remove to avoid warning:

        self._lgbm: LGBMRegressor = LGBMRegressor(
            verbosity=verbosity,
            force_row_wise=True,  # Use row-wise tree construction
            random_state=random_state,
            **kwargs,
        )

        self.species = species
        self.group_by = group_by
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.num_iter = 100
        self.verbosity = verbosity
        self.best_params_: dict[str, Any] = {}

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get the parameters of the regressor."""
        return {
            "species": self.species,
            "group_by": self.group_by,
            "cv": self.cv,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
            "verbosity": self.verbosity,
            # Include the model parameters
            **self._lgbm.get_params(deep=deep),
        }

    def set_params(self, **params: Any) -> LGBMEstimator:
        """Set the parameters of the regressor."""
        self.species = params.pop("species", self.species)
        self.group_by = params.pop("group_by", self.group_by)
        self.cv = params.pop("cv", self.cv)
        self.n_jobs = params.pop("n_jobs", self.n_jobs)
        self.random_state = params.pop("random_state", self.random_state)
        self.verbosity = params.pop("verbosity", self.verbosity)

        if self._lgbm is None:
            self._lgbm = LGBMRegressor(**params)
        else:
            self._lgbm.set_params(**params)

        return self

    def optimize_hyperparameters(
        self,
        X: MatrixLike,
        y: VectorLike,
        groups: VectorLike | None = None,
        ablation: Ablation = "all",
        use_temporal_cv: bool = False,
        use_caching: bool = True,
    ) -> tuple[dict[str, Any], float]:
        """Optimize hyperparameters for a given species.

        Parameters
        ----------
        use_caching
            Whether to use caching for the optimization, by default True.

        Returns
        -------
        A tuple containing the best hyperparameters and the best value found.
        """
        if use_temporal_cv:
            temporal_label = "with_temp_blocking"
        else:
            temporal_label = "without_temp_blocking"
        study_name = f"./cache/study-{self.species}-{self.group_by}-{ablation}-{temporal_label}.pkl"

        # Check if the study has been cached
        if use_caching and os.path.exists(study_name):
            logging.info(
                f"Loading cached study for {self.species} with group_col={self.group_by}."
            )
            study = joblib.load(study_name)
            return study.best_trial.params, study.best_value

        def objective_fn(trial: Trial) -> float:
            # See https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
            learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
            max_depth = trial.suggest_categorical(
                "max_depth", [-1, 3, 4, 5, 6, 7, 8, 9, 10, 12]
            )

            if max_depth == -1:
                num_leaves = trial.suggest_int("num_leaves", 8, 256, log=True)
            else:
                num_leaves = trial.suggest_int(
                    "num_leaves", 8, min(2**max_depth, 1024), log=True
                )

            min_child_samples = trial.suggest_int(
                "min_child_samples", 5, 1000, log=True
            )
            min_sum_hessian_in_leaf = trial.suggest_float(
                "min_sum_hessian_in_leaf", 1e-3, 10.0, log=True
            )
            lambda_l1 = trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True)
            lambda_l2 = trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True)
            min_split_gain = trial.suggest_float("min_split_gain", 0.0, 2.0)
            feature_fraction = trial.suggest_float("feature_fraction", 0.5, 1.0)
            bagging_fraction = trial.suggest_float("bagging_fraction", 0.5, 1.0)

            if bagging_fraction < 0.999:
                bagging_freq = trial.suggest_int("bagging_freq", 1, 7)
            else:
                bagging_freq = 0

            max_bin = trial.suggest_int("max_bin", 127, 511, log=True)
            extra_trees = trial.suggest_categorical("extra_trees", [False, True])
            path_smooth = trial.suggest_float("path_smooth", 0.0, 1.0)

            estimator = LGBMRegressor(
                learning_rate=learning_rate,
                max_depth=max_depth,
                num_leaves=num_leaves,
                min_child_samples=min_child_samples,
                min_sum_hessian_in_leaf=min_sum_hessian_in_leaf,
                lambda_l1=lambda_l1,
                lambda_l2=lambda_l2,
                min_split_gain=min_split_gain,
                feature_fraction=feature_fraction,
                bagging_fraction=bagging_fraction,
                bagging_freq=bagging_freq,
                max_bin=max_bin,
                extra_trees=extra_trees,
                path_smooth=path_smooth,
                boosting_type="gbdt",
                objective="regression",
                metric="rmse",
                force_row_wise=True,
                verbosity=self.verbosity,
                random_state=self.random_state,
            )

            results = cross_validate(
                estimator=estimator,
                X=to_pandas(X),
                y=to_numpy(y),
                groups=to_numpy(groups),
                scoring=make_scorer(r2_score),
                cv=GroupKFold(n_splits=self.cv),
                n_jobs=self.n_jobs,
            )

            # Rename test and train score keys to test_r2
            results["test_r2"] = results.pop("test_score")

            return results["test_r2"].mean()

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        with suppress_stderr():
            study.optimize(objective_fn, n_trials=self.num_iter)

        print(f"Best parameters found: {study.best_params}")
        print(f"with test R2: {study.best_value}")

        if use_caching:
            if not os.path.exists("./cache"):
                os.makedirs("./cache")

            joblib.dump(study, study_name)

        return study.best_trial.params, study.best_value

    def fit(self, X: MatrixLike, y: VectorLike, **kwargs: Any) -> LGBMEstimator:
        """Fit the regressor to the training data."""
        # Extract groups if provided
        groups = kwargs.get("groups", None)
        ablation = kwargs.get("ablation", "all")

        if groups is None:
            raise ValueError("Group information is required for cross-validation.")

        # Optimize hyperparameters if not already done
        best_params, _ = self.optimize_hyperparameters(
            X, y, ablation=ablation, groups=groups, use_caching=True
        )

        self.best_params_ = dict(best_params)
        self._lgbm.set_params(**best_params)
        best_params.setdefault("verbosity", self.verbosity)

        self._lgbm.fit(to_pandas(X), to_numpy(y))

        return self

    def predict(self, X: MatrixLike) -> VectorLike:
        """Predict using the fitted regressor."""
        return self._lgbm.predict(to_pandas(X))  # type: ignore[return-value]

    def get_hyperparams(self) -> dict[str, Any]:
        return dict(self.best_params_)

    def get_lgbm(self) -> LGBMRegressor:
        """Get the underlying LightGBM regressor."""
        if self._lgbm is None:
            raise ValueError("Model has not been fitted yet.")
        return self._lgbm


class MissingnessFilter:
    """Drop columns whose missing rate in the training data exceeds `threshold`."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.mask_: np.ndarray = np.array([], dtype=bool)

    def fit(self, X: np.ndarray, y: Any = None) -> MissingnessFilter:
        self.mask_ = np.isnan(X).mean(axis=0) <= self.threshold
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.mask_]

    def fit_transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        return self.fit(X).transform(X)

    def get_support(self) -> np.ndarray:
        return self.mask_


class ElasticNetEstimator(EstimatorProtocol):
    """ElasticNet regressor with cross-validated hyperparameter selection."""

    def __init__(
        self,
        *,
        species: Species,
        group_by: str,
        cv: int = 5,
        random_state: int | None = RANDOM_STATE,
        **kwargs: Any,
    ):
        """Initialize the ElasticNet regressor."""
        self.species = species
        self.group_by = group_by
        self.cv = cv
        self.random_state = random_state
        self.elasticnet_kwargs = kwargs.copy()

        self._miss_filter: MissingnessFilter | None = None
        self._preprocessor = None
        self._model = None
        self._var_mask: np.ndarray | None = None
        self._y_min: float = -np.inf
        self._y_max: float = np.inf

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get the parameters of the regressor."""
        if self._model is None:
            raise ValueError("Model has not been fitted yet.")

        return self._model.get_params(deep=deep)

    def set_params(self, **params: Any) -> ElasticNetEstimator:
        """Set the parameters of the regressor."""
        if self._model is None:
            raise ValueError("Model has not been fitted yet.")

        self._model.set_params(**params)
        return self

    def fit(self, X: MatrixLike, y: VectorLike, **kwargs: Any) -> ElasticNetEstimator:
        """Fit the regressor to the training data."""
        groups = kwargs.get("groups", None)
        fold: int | None = kwargs.get("fold", None)
        feature_names = list(X.columns) if isinstance(X, pl.DataFrame) else None
        tag = f"{self.species}|fold={fold}" if fold is not None else self.species

        X_np = to_numpy(X).astype(float)
        y_arr = to_numpy(y).astype(float)
        self._y_min, self._y_max = float(y_arr.min()), float(y_arr.max())

        # Log per-feature missing rates (before imputation)
        if feature_names is not None:
            miss_rate = np.isnan(X_np).mean(axis=0)
            high_miss = [
                (f, float(r)) for f, r in zip(feature_names, miss_rate) if r > 0.5
            ]
            if high_miss:
                logging.info(
                    "[%s] %d feature(s) with >50%% missing: %s",
                    tag,
                    len(high_miss),
                    ", ".join(f"{f}={r:.0%}" for f, r in high_miss),
                )

        # Step 1: drop features with >50% missing *before* imputation.
        # VarianceThreshold alone won't catch these: a feature that is 80% missing
        # still has real variance in the observed 20%, survives the threshold, then
        # gets imputed to a training-fold constant — causing test-time distribution
        # shift when the temporal test fold has different missingness patterns.
        self._miss_filter = MissingnessFilter(threshold=0.5)
        X_np_filtered = self._miss_filter.fit_transform(X_np)
        miss_mask = self._miss_filter.get_support()
        if feature_names is not None:
            dropped_miss = [f for f, keep in zip(feature_names, miss_mask) if not keep]
            if dropped_miss:
                logging.info(
                    "[%s] Dropped %d high-missingness feature(s) (>50%%): %s",
                    tag,
                    len(dropped_miss),
                    dropped_miss,
                )

        # Step 2: impute → drop near-zero variance → standardise on the filtered matrix
        self._preprocessor = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("var_threshold", VarianceThreshold(threshold=1e-4)),
                ("scaler", RobustScaler()),
            ]
        )

        self._preprocessor.fit(X_np_filtered)
        X_proc = self._preprocessor.transform(X_np_filtered).astype(float)

        # Combine both masks so _var_mask indexes into the *original* feature space
        var_mask_partial = self._preprocessor.named_steps["var_threshold"].get_support()
        full_mask = np.zeros(X_np.shape[1], dtype=bool)
        full_mask[miss_mask] = var_mask_partial
        self._var_mask = full_mask
        if feature_names is not None:
            kept_after_miss = [f for f, keep in zip(feature_names, miss_mask) if keep]
            dropped_var = [
                f for f, keep in zip(kept_after_miss, var_mask_partial) if not keep
            ]
            if dropped_var:
                logging.info(
                    "[%s] Dropped %d near-zero variance feature(s): %s",
                    tag,
                    len(dropped_var),
                    dropped_var,
                )

        # Eigenvalues of the Gram matrix (p×p) for condition diagnostics and α grid.
        n_obs, _p = X_proc.shape
        G = X_proc.T @ X_proc
        eigvals = np.linalg.eigvalsh(G)  # ascending, real (symmetric matrix)
        eig_min = max(float(eigvals[0]), 0.0)
        eig_max = float(eigvals[-1])
        cond_gram = eig_max / eig_min if eig_min > 0.0 else np.inf
        logging.info(
            "[%s] Gram matrix: shape=(%d×%d), eig_min=%.3e, eig_max=%.3e, cond=%.2e",
            tag,
            _p,
            _p,
            eig_min,
            eig_max,
            cond_gram,
        )

        # Data-driven α grid -------------------------------------------------------
        # Upper bound: α at which all coefficients vanish (sklearn path convention),
        #   α_max(λ) = max|X^T y| / (n·λ).  We use the smallest l1_ratio so the grid
        #   covers the full useful range for all mixing ratios.
        # Lower bound: smallest α s.t. cond(G + n·α·(1−λ)·I) ≤ COND_MAX.
        #   Solving (eig_max + n·α·(1−λ)) / (eig_min + n·α·(1−λ)) = COND_MAX gives:
        #     α_floor = (eig_max − COND_MAX·eig_min) / (n·(1−λ)·(COND_MAX−1))
        #   Worst case: largest l1_ratio (smallest ridge term 1−λ), requiring highest α.
        _L1_RATIOS = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99]
        COND_MAX = 1e4

        alpha_max = float(np.max(np.abs(X_proc.T @ y_arr))) / n_obs

        _worst_l1 = max(lr for lr in _L1_RATIOS if lr < 1.0)  # 0.99
        _excess = eig_max - COND_MAX * eig_min
        alpha_cond_floor = (
            _excess / (n_obs * (1.0 - _worst_l1) * (COND_MAX - 1))
            if _excess > 0.0
            else 0.0
        )
        # Hard fallback matches sklearn's default eps=1e-3 (alpha_min = alpha_max / 1000).
        # A smaller value risks near-unregularised CD paths that never converge.
        alpha_min = max(alpha_cond_floor, alpha_max * 1e-3)

        if alpha_min >= alpha_max:
            logging.warning(
                "[%s] Condition-number floor (%.3e) >= alpha_max (%.3e); "
                "falling back to alpha_max * 1e-3",
                tag,
                alpha_cond_floor,
                alpha_max,
            )
            alpha_min = alpha_max * 1e-3

        logging.info(
            "[%s] alpha grid: alpha_max=%.3e, cond_floor=%.3e, alpha_min=%.3e (100 pts)",
            tag,
            alpha_max,
            alpha_cond_floor,
            alpha_min,
        )
        alphas_grid = np.logspace(np.log10(alpha_min), np.log10(alpha_max), 100)
        # --------------------------------------------------------------------------

        # Pre-compute splits so ElasticNetCV.fit() never needs a groups= kwarg
        groups_arr = to_numpy(groups)
        splitter = GroupKFold(n_splits=self.cv)
        cv_splits = list(splitter.split(X_proc, y_arr, groups=groups_arr))

        en_cv = ElasticNetCV(
            l1_ratio=_L1_RATIOS,
            alphas=alphas_grid,
            cv=cv_splits,
            max_iter=10_000,
            tol=1e-3,
            random_state=self.random_state,
            verbose=False,
        )
        en_cv.fit(X_proc, y_arr)
        alpha, l1_ratio = en_cv.alpha_, en_cv.l1_ratio_

        if alpha <= alpha_min * 1.5:
            logging.warning(
                "[%s] Selected alpha (%.3e) is at/near the grid floor (%.3e); "
                "path may be under-regularised — consider raising COND_MAX",
                tag,
                alpha,
                alpha_min,
            )

        # Log per-fold CV MSE at the selected (alpha, l1_ratio) to spot bad folds
        l1_idx = int(np.argmin(np.abs(np.atleast_1d(en_cv.l1_ratio) - l1_ratio)))
        alpha_idx = int(np.argmin(np.abs(en_cv.alphas_[l1_idx] - alpha)))
        fold_mses = en_cv.mse_path_[l1_idx][alpha_idx]
        logging.info(
            "[%s] CV selected alpha=%.6f, l1_ratio=%.2f | fold MSEs: %s",
            tag,
            alpha,
            l1_ratio,
            " ".join(f"{v:.4f}" for v in fold_mses),
        )

        en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=100_000)
        en.fit(X_proc, y_arr)

        coef = en.coef_
        n_nonzero = int(np.sum(np.abs(coef) > 1e-6))
        max_coef = float(np.max(np.abs(coef)))
        if feature_names is not None:
            kept_names = [f for f, keep in zip(feature_names, self._var_mask) if keep]
            top_idx = np.argsort(np.abs(coef))[::-1][:5]
            top = [(kept_names[i], float(coef[i])) for i in top_idx]
            top_str = ", ".join(f"{f}={v:+.4f}" for f, v in top)
        else:
            top_str = "(feature names unavailable)"
        logging.info(
            "[%s] Coefficients: n_nonzero=%d, max|coef|=%.4f | top-5: %s",
            tag,
            n_nonzero,
            max_coef,
            top_str,
        )

        self._model = en

        return self

    def predict(self, X: MatrixLike) -> VectorLike:
        """Predict using the fitted regressor."""
        if self._model is None:
            raise ValueError("Model has not been fitted yet.")
        return np.clip(self._model.predict(self.transform(X)), self._y_min, self._y_max)

    def transform(self, X: MatrixLike) -> np.ndarray:
        """Apply the full preprocessing stack (missingness filter → impute → variance filter → scale)."""
        if self._preprocessor is None or self._miss_filter is None:
            raise ValueError("Model has not been fitted yet.")
        X_np = to_numpy(X).astype(float)
        return self._preprocessor.transform(self._miss_filter.transform(X_np))

    def get_hyperparams(self) -> dict[str, Any]:
        if self._model is None:
            raise ValueError("Model has not been fitted yet.")
        return {
            "alpha": float(self._model.alpha),
            "l1_ratio": float(self._model.l1_ratio),
        }

    def get_sklearn(self) -> ElasticNet:
        """Get the underlying ElasticNet regressor."""
        if self._model is None:
            raise ValueError("Model has not been fitted yet.")

        return self._model


@dataclass
class _FixedEffectLinear:
    """Minimal sklearn-compatible wrapper for SHAP LinearExplainer (fixed effects only)."""

    coef_: np.ndarray
    intercept_: float


class MixedLMEstimator(EstimatorProtocol):
    """Mixed-effects linear model with a per-plot random intercept.

    Uses the same preprocessing pipeline as ElasticNetEstimator (missingness
    filter → median imputation → near-zero variance filter → RobustScaler).
    The model is fitted with REML, which gives unbiased variance-component
    estimates and therefore more accurate BLUPs.

    Prediction convention for unseen plots
    ---------------------------------------
    At test time the random intercept for an unseen plot is set to zero
    (population-level prediction).  This is consistent with how GBDT and
    ElasticNet are evaluated: both ignore plot identity at prediction time and
    are therefore assessed on the same basis as the fixed-effects-only part of
    the LMM.
    """

    def __init__(
        self,
        *,
        species: Species,
        group_by: str,
        reml: bool = True,
        **kwargs: Any,
    ) -> None:
        self.species = species
        self.group_by = group_by
        self.reml = reml

        self._miss_filter: MissingnessFilter | None = None
        self._preprocessor: Pipeline | None = None
        self._result: Any = None  # statsmodels MixedLMResultsWrapper
        self._var_mask: np.ndarray | None = None
        self._feature_names_proc: list[str] = []

        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._y_min: float = -np.inf
        self._y_max: float = np.inf

        self._converged: bool = True
        self.var_random: float | None = None
        self.var_resid: float | None = None
        self.icc: float | None = None
        self.plot_blup_: dict[str, float] = {}

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        if self._result is None:
            raise ValueError("Model has not been fitted yet.")
        return {"reml": self.reml}

    def set_params(self, **params: Any) -> "MixedLMEstimator":
        if "reml" in params:
            self.reml = params["reml"]
        return self

    def fit(self, X: MatrixLike, y: VectorLike, **kwargs: Any) -> "MixedLMEstimator":
        import pandas as pd
        import statsmodels.api as sm
        from statsmodels.regression.mixed_linear_model import MixedLM

        plot_groups = kwargs.get("plot_groups", None)
        fold: int | None = kwargs.get("fold", None)
        feature_names = list(X.columns) if isinstance(X, pl.DataFrame) else None
        tag = f"{self.species}|fold={fold}" if fold is not None else self.species

        X_np = to_numpy(X).astype(float)
        y_arr = to_numpy(y).astype(float)

        # Standardise target to improve optimiser convergence
        self._y_mean = float(y_arr.mean())
        self._y_std = float(y_arr.std()) or 1.0
        y_std = (y_arr - self._y_mean) / self._y_std
        self._y_min, self._y_max = float(y_arr.min()), float(y_arr.max())

        # --- Preprocessing: identical to ElasticNetEstimator ---
        self._miss_filter = MissingnessFilter(threshold=0.5)
        X_np_filtered = self._miss_filter.fit_transform(X_np)
        miss_mask = self._miss_filter.get_support()

        self._preprocessor = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("var_threshold", VarianceThreshold(threshold=1e-4)),
                ("scaler", RobustScaler()),
            ]
        )
        self._preprocessor.fit(X_np_filtered)
        X_proc = self._preprocessor.transform(X_np_filtered).astype(float)

        var_mask_partial = self._preprocessor.named_steps["var_threshold"].get_support()
        full_mask = np.zeros(X_np.shape[1], dtype=bool)
        full_mask[miss_mask] = var_mask_partial
        self._var_mask = full_mask

        if feature_names is not None:
            kept_after_miss = [f for f, keep in zip(feature_names, miss_mask) if keep]
            self._feature_names_proc = [
                f for f, keep in zip(kept_after_miss, var_mask_partial) if keep
            ]
        else:
            self._feature_names_proc = [f"x{i}" for i in range(X_proc.shape[1])]

        # --- Fit MixedLM with random intercept per plot ---
        X_pd = pd.DataFrame(X_proc, columns=self._feature_names_proc)
        X_pd_const = sm.add_constant(X_pd, has_constant="add")
        y_pd = pd.Series(y_std, name="y")

        if plot_groups is not None:
            groups_series = pd.Series(plot_groups.astype(str), name="plot_id")
        else:
            logging.warning(
                "[%s] No plot_groups; random intercept will be degenerate", tag
            )
            groups_series = pd.Series(
                np.zeros(len(y_pd), dtype=int).astype(str), name="plot_id"
            )

        model = MixedLM(endog=y_pd, exog=X_pd_const, groups=groups_series)

        result = None
        for method in ("lbfgs", "bfgs", "nm"):
            try:
                result = model.fit(reml=self.reml, method=method, disp=False)
                self._converged = bool(result.converged)
                if self._converged:
                    break
                logging.warning(
                    "[%s] MixedLM method=%s did not converge; trying next", tag, method
                )
            except Exception as exc:
                logging.warning("[%s] MixedLM method=%s failed: %s", tag, method, exc)

        if result is None:
            raise RuntimeError(f"[{tag}] MixedLM failed with all optimisation methods")

        if not self._converged:
            logging.warning(
                "[%s] MixedLM: no method converged; variance estimates may be unreliable",
                tag,
            )

        self.var_random = float(result.cov_re.iloc[0, 0])
        self.var_resid = float(result.scale)
        total_var = self.var_random + self.var_resid
        self.icc = self.var_random / total_var if total_var > 0 else 0.0

        logging.info(
            "[%s] MixedLM: var_random=%.4f, var_resid=%.4f, ICC=%.4f, converged=%s",
            tag,
            self.var_random,
            self.var_resid,
            self.icc,
            self._converged,
        )

        top_idx = np.argsort(np.abs(result.fe_params.values))[::-1][:5]
        top_str = ", ".join(
            f"{result.fe_params.index[i]}={result.fe_params.iloc[i]:+.4f}"
            for i in top_idx
        )
        logging.info("[%s] Top-5 fixed effects: %s", tag, top_str)

        # BLUPs (in standardised target space): û_j from statsmodels random_effects
        self.plot_blup_ = {
            str(plot_id): float(re.iloc[0])
            for plot_id, re in result.random_effects.items()
        }
        logging.info(
            "[%s] BLUP: stored random intercepts for %d training plots",
            tag,
            len(self.plot_blup_),
        )

        self._result = result
        return self

    def predict(
        self, X: MatrixLike, plot_groups: np.ndarray | None = None
    ) -> np.ndarray:
        """Predict using fixed effects plus optional per-plot BLUP adjustment.

        Parameters
        ----------
        plot_groups
            Plot identifier for each row.  When provided, the BLUP û_j estimated
            during training is added for every plot seen in the training fold.
            Observations whose plot was not in training receive û_j = 0
            (population-level prediction).  Pass None to always use fixed effects
            only (e.g. for SHAP attribution, which must reconstruct y_pred
            without the random-intercept term).
        """
        if self._result is None:
            raise ValueError("Model has not been fitted yet.")
        X_proc = self.transform(X)
        X_with_const = np.column_stack([np.ones(len(X_proc)), X_proc])
        y_std_pred = X_with_const @ self._result.fe_params.values

        if plot_groups is not None and self.plot_blup_:
            blup_adj = np.fromiter(
                (self.plot_blup_.get(str(p), 0.0) for p in plot_groups),
                dtype=float,
                count=len(plot_groups),
            )
            y_std_pred = y_std_pred + blup_adj

        return np.clip(
            y_std_pred * self._y_std + self._y_mean, self._y_min, self._y_max
        )

    def transform(self, X: MatrixLike) -> np.ndarray:
        """Apply the preprocessing stack (missingness filter → impute → variance filter → scale)."""
        if self._preprocessor is None or self._miss_filter is None:
            raise ValueError("Model has not been fitted yet.")
        X_np = to_numpy(X).astype(float)
        return self._preprocessor.transform(self._miss_filter.transform(X_np))

    def get_hyperparams(self) -> dict[str, Any]:
        if self._result is None:
            raise ValueError("Model has not been fitted yet.")
        return {
            "var_random": float(self.var_random or 0.0),
            "var_resid": float(self.var_resid or 0.0),
            "icc": float(self.icc or 0.0),
            "converged": self._converged,
        }

    def get_linear_model(self) -> _FixedEffectLinear:
        """Return a sklearn-compatible wrapper around the fixed effects for SHAP."""
        if self._result is None:
            raise ValueError("Model has not been fitted yet.")
        fe = self._result.fe_params.values
        # Rescale from standardised target space back to original target space
        coef = fe[1:] * self._y_std
        intercept = float(fe[0] * self._y_std + self._y_mean)
        return _FixedEffectLinear(coef_=coef, intercept_=intercept)


@dataclass
class ExperimentResults:
    """Results of an experiment.

    Attributes
    ----------
    species
        Species for which the experiment was run.
    ablation
        Ablation study performed on the model.
    temporal_blocking
        Whether temporal blocking was used.
    X
        Dataframe containing the features.
    metadata
        Dataframe containing metadata columns (non-feature columns).
    y_true
        True target values.
    y_pred
        Predicted target values (one series per fold).
    indices
        Indices for the training and test sets.
    estimators
        Trained estimators.
    explainers
        SHAP explainers.
    performances
        Performance metrics (one dictionary per fold).
    shap_values
        SHAP values (one Explanation object per fold).
    """

    species: Species
    ablation: Ablation
    temporal_blocking: bool

    X: pl.DataFrame
    metadata: pl.DataFrame
    y_true: pl.Series
    y_pred: Sequence[pl.Series]

    indices: Sequence[dict[Split, np.ndarray]]
    estimators: Sequence[EstimatorProtocol]
    explainers: Sequence[Explainer]

    performances: Sequence[dict[str, float]]

    shap_values: Sequence[Explanation]
    # Required for correct fold-to-SHAP alignment; adding this field is a breaking
    # change — cached ExperimentResults pickled before this field was added must be
    # regenerated (clear cache/ and re-run train-all.sh).
    shap_row_indices: Sequence[np.ndarray]
    dist_params: tuple[float | None, float | None, float | None] | None = None

    @property
    def num_folds(self) -> int:
        return len(self.y_pred)

    @property
    def features(self) -> list[str]:
        return self.X.columns

    def get_indices(self, fold: int, split: Split) -> np.ndarray:
        """Get indices for the given fold and split."""
        if split == "all":
            return np.concatenate(
                [self.indices[fold]["train"], self.indices[fold]["test"]]
            )
        else:
            return self.indices[fold][split]

    def get_data(
        self, fold: int, split: Split
    ) -> tuple[pl.DataFrame, pl.Series, pl.Series]:
        """Get training data for the given fold.

        Parameters
        ----------
        fold
            Fold index.
        split
            Split type (train or test).

        Returns
        -------
        A tuple (X, y_true, y_pred) containing the data for the given fold and split.
        """
        indices = self.get_indices(fold, split)

        return (
            self.X[indices],
            self.y_true[indices],
            self.y_pred[fold][indices],
        )

    def get_shap_values(self, fold: int, split: Split = "test") -> Explanation:
        """Get SHAP values for the given fold.

        Parameters
        ----------
        fold
            Fold index.
        split
            Split type (train or test).

        Returns
        -------
        SHAP values for the given fold and split.
        """
        shap_values = self.shap_values[fold]

        if shap_values is None:
            raise ValueError(
                f"No SHAP values available for fold {fold}. "
                "Ensure that the model was trained with SHAP explanations."
            )
        used_idx = (
            self.shap_row_indices[fold]
            if fold < len(self.shap_row_indices)
            else np.arange(len(self.X))
        )
        indices = self.get_indices(fold, split)
        mask = np.isin(used_idx, indices)

        return cast(Explanation, shap_values[mask])

    def get_shap_interactions(
        self, fold: int, split: Split = "test", num_samples: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get SHAP interaction values for the given fold.

        Note: this is only available for tree-based models (e.g., LightGBM).

        Parameters
        ----------
        fold
            Fold index.
        split
            Split type ('train', 'test', or 'all').
        num_samples
            Number of samples to use for the SHAP interaction values (None for all samples).

        Returns
        -------
        A tuple (interactions, indices) containing the interaction values and the indices of the
        features.
        """
        explainer = self.explainers[fold]

        if explainer is None:
            raise ValueError(
                f"No SHAP explainer available for fold {fold}. "
                "Ensure that the model was trained with SHAP explanations."
            )

        if not isinstance(explainer, TreeExplainer):
            raise ValueError(
                "SHAP interaction values are only available for tree-based models. "
            )

        indices = self.get_indices(fold, split)

        if num_samples is not None and num_samples < len(indices):
            indices = np.random.choice(indices, num_samples, replace=False)

        interactions = cast(
            np.ndarray,
            explainer.shap_interaction_values(self.X[indices].to_numpy()),
        )

        return interactions, indices

    def get_inverse_transform(
        self, y: pl.Series, dist_params: tuple[float | None, float | None, float | None]
    ) -> pl.Series:
        """
        Get the inverse transform for the given y.

        This is used to transform the y values back to the original scale of the target variable.

        Parameters
        ----------
        y
            The y values to transform.
        dist_params
            A tuple containing the shape, location, and scale parameters of the log-normal distribution.

        Returns
        -------
        The inverse transform for the given y.
        """
        shape, loc, scale = dist_params
        if shape is None or loc is None or scale is None:
            raise ValueError("dist_params contains None; cannot inverse transform.")
        # Clip away exact 0/1 to avoid lognorm.ppf returning -inf/+inf at the CDF boundaries.
        y_orig = pl.Series(
            lognorm.ppf(
                np.clip(to_numpy(y), 1e-6, 1 - 1e-6),
                s=shape,
                loc=loc,
                scale=scale,
            )
            - 1
        )

        return y_orig

    def get_shap_values_orig_space(
        self, fold: int, split: Split = "test"
    ) -> pl.DataFrame:
        """
        Return per-feature SHAP attributions in original growth-rate space (% yr⁻¹).

        The model is trained on PIT-quantile targets ỹ = F_lognorm(log-RGR + 1),
        so raw SHAP values φᵢ are expressed in quantile units.  A naive
        back-transformation via the inverse PIT f⁻¹ is prediction-point dependent
        and asymmetric.  We therefore use a symmetric finite difference:

            δᵢ = f⁻¹(ŷ + φᵢ/2) − f⁻¹(ŷ − φᵢ/2)

        This is a second-order accurate approximation of the true attribution in
        original space, centred on the predicted quantile ŷ.  It coincides with
        the first-order (asymmetric) estimator when φᵢ is small, and reduces
        curvature bias when φᵢ is large (e.g. defoliation for spruce).

        The base value in original space is set to f⁻¹(base_value), consistent
        with the additivity property:

            f⁻¹(base) + Σᵢ δᵢ ≈ f⁻¹(ŷ)    [to second order in φᵢ]

        Note: exact additivity does not hold after the nonlinear back-transform;
        the residual is O(φᵢ² · (f⁻¹)''(ŷ)) and is negligible for small SHAP
        values but may be non-trivial for dominant features.

        Parameters
        ----------
        fold
            Fold index.
        split
            Split type ('train', 'test', or 'all').

        Returns
        -------
        pl.DataFrame
            SHAP attributions in original space, shape (n_samples, n_features).
        """
        if self.dist_params is None:
            raise ValueError(
                "dist_params is None. Cannot compute attributions in original space."
            )

        # --- 1. Retrieve quantile-space predictions and SHAP values --------------
        X, y_true, y_pred_series = self.get_data(fold, split)
        y_pred = y_pred_series.to_numpy()

        used_idx = (
            self.shap_row_indices[fold]
            if fold < len(self.shap_row_indices)
            else np.arange(len(self.X))
        )
        indices = self.get_indices(fold, split)
        mask = np.isin(used_idx, indices)

        shap_values = np.asarray(
            self.shap_values[fold].values[mask]
        )  # (n_samples, n_features)
        base_values = np.asarray(
            self.shap_values[fold].base_values[mask]
        )  # (n_samples,)

        # --- 2. Verify SHAP additivity in quantile space -------------------------
        u_pred_reconstructed = base_values + shap_values.sum(axis=1)
        if not np.allclose(u_pred_reconstructed, y_pred, rtol=1e-6, atol=1e-6):
            max_abs_err = np.max(np.abs(u_pred_reconstructed - y_pred))
            mean_abs_err = np.mean(np.abs(u_pred_reconstructed - y_pred))
            logging.warning(
                f"{self.species} fold {fold}: SHAP additivity check failed "
                f"(max_abs_err={max_abs_err:.6g}, mean_abs_err={mean_abs_err:.6g})"
            )

        # --- 3. Symmetric finite difference in quantile space --------------------
        # u_plus[j, i]  = ŷⱼ + φᵢⱼ / 2
        # u_minus[j, i] = ŷⱼ − φᵢⱼ / 2
        u_pred = y_pred  # shape: (n_samples,)
        half_shap = shap_values / 2.0  # (n_samples, n_features)
        u_plus = u_pred[:, None] + half_shap  # (n_samples, n_features)
        u_minus = u_pred[:, None] - half_shap  # (n_samples, n_features)

        # --- 4. Apply inverse PIT to both endpoints ------------------------------
        n_samples, n_features = shap_values.shape

        dist_params = self.dist_params

        def inv_transform_matrix(u_matrix: np.ndarray) -> np.ndarray:
            flat = pl.Series(u_matrix.ravel())
            return to_numpy(self.get_inverse_transform(flat, dist_params)).reshape(
                n_samples, n_features
            )

        y_plus = inv_transform_matrix(u_plus)  # (n_samples, n_features)
        y_minus = inv_transform_matrix(u_minus)  # (n_samples, n_features)

        # --- 5. Symmetric attribution in original space --------------------------
        delta = y_plus - y_minus  # (n_samples, n_features)

        return pl.DataFrame(delta, schema=self.features)


@dataclass
class CrossValidationResults:
    test_r2: list[float] = field(default_factory=list)
    train_r2: list[float] = field(default_factory=list)
    test_rmse: list[float] = field(default_factory=list)
    train_rmse: list[float] = field(default_factory=list)
    estimator: list[EstimatorProtocol] = field(default_factory=list)
    indices: dict[Split, list[pl.Series]] = field(
        default_factory=lambda: {"train": [], "test": []}
    )


def train_and_explain(
    species: Species,
    *,
    model_type: ModelType,
    ablation: Ablation = "all",
    group_by: str,
    cv: int = 5,
    n_jobs: int = -1,
    use_temporal_cv: bool = False,
) -> ExperimentResults:
    """Train models for the given species.

    Parameters
    ----------
    species
        Species to train the model for.
    model_type
        Type of model to use for training, either "gbdt" or "elasticnet".
    group_by
        Column to group by for cross-validation.
    cv
        Number of cross-validation folds, by default 5.
    n_jobs
        Number of jobs to run in parallel, by default -1.

    Returns
    -------
    A `Result` object containing the trained models and SHAP values.
    """
    print(f"Training model for {species}")

    # Load data for the given species
    df = load_data(species)

    # Prepare data
    X, y, dist_params = prepare_data(df, ablation)

    if group_by == "plot_id":
        use_temporal_cv = (
            False  # Temporal CV is not compatible with plot-level grouping
        )

    # Prepare groups
    groups = df.select(group_by).to_series()
    plot_groups = df.select("plot_id").to_series()

    # Use Hierarchical Temporal Group CV to remove temporal autocorrelation in the splits
    if use_temporal_cv:
        from HierarchicalTemporalGroupCV import HierarchicalTimeGroupCV

        temporal_cv = HierarchicalTimeGroupCV(
            log_level=logging.ERROR,
            random_state=RANDOM_STATE,
        )
        splits = []
        for fold, (train_idx, test_idx) in enumerate(
            temporal_cv.run_cross_validation(
                species=species,
                ablation=ablation,
                tree_group=group_by,
            )
        ):
            splits.append((train_idx, test_idx))
    else:
        splits = []
        for fold, (train_idx, test_idx) in enumerate(
            GroupKFold(n_splits=cv).split(to_numpy(X), y, groups=to_numpy(groups))
        ):
            splits.append((train_idx, test_idx))

    # Cross-validation loop
    print(f"Starting cross-validation for {species} with {model_type} estimator...")

    results = CrossValidationResults()

    for fold, (train_idx, test_idx) in enumerate(splits):
        # Split data into training and test sets
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Create estimator
        if model_type == "gbdt":
            sklearn.set_config(enable_metadata_routing=False)

            estimator = LGBMEstimator(
                species=species,
                group_by=group_by,
                cv=cv,
                n_jobs=n_jobs,
            )
        elif model_type == "elasticnet":
            estimator = ElasticNetEstimator(
                species=species,
                group_by=group_by,
                cv=cv,
            )
        elif model_type == "lmm":
            estimator = MixedLMEstimator(
                species=species,
                group_by=group_by,
            )
        else:
            raise ValueError(
                f"Unknown estimator: {model_type}. Supported: 'gbdt', 'elasticnet', 'lmm'."
            )

        print(f"Fold {fold + 1}/{cv}")

        # Fit the model
        estimator.fit(
            X_train,
            y_train,
            groups=to_numpy(groups[train_idx]) if groups is not None else None,
            plot_groups=to_numpy(plot_groups[train_idx]),
            ablation=ablation,
            fold=fold,
        )

        # Evaluate the model.
        # For LMM with tree-wise CV the same plot can appear in both train and
        # test (different trees), so we add the per-plot BLUP to predictions.
        # SHAP and y_pred storage always use fixed effects only (predict without
        # plot_groups) so that LinearExplainer attribution remains consistent.
        if isinstance(estimator, MixedLMEstimator) and group_by == "tree_id":
            _pg_train = to_numpy(plot_groups[train_idx])
            _pg_test = to_numpy(plot_groups[test_idx])
            _yhat_train = estimator.predict(X_train, plot_groups=_pg_train)
            _yhat_test = estimator.predict(X_test, plot_groups=_pg_test)
            r2_train = r2_score(y_train, _yhat_train)
            r2_test = r2_score(y_test, _yhat_test)
            rmse_train = float(root_mean_squared_error(to_numpy(y_train), _yhat_train))
            rmse_test = float(root_mean_squared_error(to_numpy(y_test), _yhat_test))
        else:
            r2_train = estimator.score(X_train, y_train)
            r2_test = estimator.score(X_test, y_test)
            rmse_train = estimator.rmse(X_train, y_train)
            rmse_test = estimator.rmse(X_test, y_test)

        # Update cross-validation results
        results.test_r2.append(r2_test)
        results.train_r2.append(r2_train)
        results.test_rmse.append(rmse_test)
        results.train_rmse.append(rmse_train)
        results.estimator.append(estimator)
        results.indices["train"].append(pl.Series("train_idx", train_idx))
        results.indices["test"].append(pl.Series("test_idx", test_idx))

        # Print R2 score for the fold
        print(
            f"Fold {fold + 1}: R2 (train) = {r2_train:.2f}, R2 (test) = {r2_test:.2f}"
        )

    print(f"Cross-validation completed for {species} with {model_type} estimator.")

    print("Summary of results:")
    print(
        f" `- R2 (test): {np.mean(results.test_r2):.2f} +/- {np.std(results.test_r2):.2f}"
    )
    print(
        f" `- R2 (train): {np.mean(results.train_r2):.2f} +/- {np.std(results.train_r2):.2f}"
    )

    # Create SHAP explainers for the trained models
    explainers = []
    shap_values = []
    shap_row_indices = []

    X_background = X.sample(1000, with_replacement=False)

    for fold, estimator in enumerate(results.estimator):
        train_idx = results.indices["train"][fold].to_numpy()
        test_idx = results.indices["test"][fold].to_numpy()
        # Create a SHAP explainer for the LGBM model
        if isinstance(estimator, LGBMEstimator):
            explainer = TreeExplainer(
                estimator.get_lgbm(),
                feature_names=X.columns,
                feature_perturbation="tree_path_dependent",
            )
        elif isinstance(estimator, (ElasticNetEstimator, MixedLMEstimator)):
            X_bg_proc = estimator.transform(X_background)
            X_proc = estimator.transform(X)
            mask = estimator._var_mask
            feat_names_proc = (
                [f for f, keep in zip(X.columns, mask) if keep]
                if mask is not None
                else X.columns
            )
            linear_model = (
                estimator.get_sklearn()
                if isinstance(estimator, ElasticNetEstimator)
                else estimator.get_linear_model()
            )
            explainer = LinearExplainer(
                linear_model,
                feature_names=feat_names_proc,
                masker=IndependentMasker(X_bg_proc),
            )
            raw = explainer(X_proc)
            assert isinstance(raw, Explanation)
            # Pad SHAP values back to the original feature space (zeros for dropped features)
            n_orig = len(X.columns)
            padded = np.zeros((len(X), n_orig))
            if mask is not None:
                padded[:, mask] = raw.values
            else:
                padded = raw.values
            shap_values.append(
                Explanation(
                    padded,
                    base_values=raw.base_values,
                    data=X.to_numpy(),
                    feature_names=list(X.columns),
                )
            )
        else:
            raise ValueError(
                f"Unsupported estimator type: {type(estimator)}. "
                "Supported types are LGBMEstimator, ElasticNetEstimator, MixedLMEstimator."
            )

        explainers.append(explainer)
        shap_row_indices.append(np.arange(len(X)))
        if not isinstance(estimator, (ElasticNetEstimator, MixedLMEstimator)):
            shap_values.append(explainer(X.to_numpy()))

    return ExperimentResults(
        species=species,
        ablation=ablation,
        temporal_blocking=use_temporal_cv,
        X=X,
        metadata=df.select(pl.selectors.exclude(*X.columns)),
        y_true=y,
        y_pred=[pl.Series("y_pred", model.predict(X)) for model in results.estimator],
        indices=[
            {
                "train": results.indices["train"][fold].to_numpy(),
                "test": results.indices["test"][fold].to_numpy(),
            }
            for fold in range(cv)
        ],
        estimators=results.estimator,
        explainers=explainers,
        performances=[
            {
                "test_r2": float(results.test_r2[fold]),
                "train_r2": float(results.train_r2[fold]),
                "test_rmse": float(results.test_rmse[fold]),
                "train_rmse": float(results.train_rmse[fold]),
                "n_train": len(results.indices["train"][fold]),
                "n_test": len(results.indices["test"][fold]),
            }
            for fold in range(cv)
        ],
        shap_values=shap_values,
        dist_params=dist_params,
        shap_row_indices=shap_row_indices,
    )
