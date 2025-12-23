# Train models
from __future__ import annotations

import sklearn.preprocessing

from config import Ablation, Species
from lightgbm import LGBMRegressor
from sklearn.linear_model import LassoCV, Lasso

import sklearn
from sklearn.model_selection import KFold, GroupKFold, cross_validate
from sklearn.metrics import mean_squared_error, make_scorer
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

Split = Literal["train", "test", "all"]
ModelType = Literal["gbdt", "lasso"]
MatrixLike = np.ndarray | pl.DataFrame
VectorLike = np.ndarray | pl.Series

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

RANDOM_STATE = 42  # Global random state for reproducibility
ALL_SPECIES: list[Species] = ["spruce", "pine", "beech", "oak"]

np.random.seed(RANDOM_STATE)  # Set the global random seed for NumPy


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
        return data.to_numpy()
    elif isinstance(data, pl.Series):
        return data.to_numpy()
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise TypeError(
            f"Unsupported data type: {type(data)}. Expected DataFrame or Series."
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


class LGBMEstimator(EstimatorProtocol):
    """LightGBM regressor."""

    def __init__(
        self,
        *,
        species: Species,
        group_by: str | None,
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
        study_name = f"./cache/study-{self.species}-{self.group_by}-{ablation}.pkl"

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

            params = dict(
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
            )

            estimator = LGBMRegressor(
                **params,  # type: ignore[arg-type]
                force_row_wise=True,
                verbosity=self.verbosity,
                random_state=self.random_state,
            )

            results = cross_validate(
                estimator=estimator,
                X=to_numpy(X),
                y=to_numpy(y),
                groups=to_numpy(groups),
                scoring=make_scorer(r2_score),
                cv=KFold(n_splits=self.cv)
                if self.group_by is None
                else GroupKFold(n_splits=self.cv),
                n_jobs=self.n_jobs,
            )

            # Rename test and train score keys to test_r2
            results["test_r2"] = results.pop("test_score")

            return results["test_r2"].mean()

        study = optuna.create_study(direction="maximize")
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

        if self.group_by is not None and groups is None:
            raise ValueError(
                "Group information is required for cross-validation with group_by."
            )

        # Optimize hyperparameters if not already done
        best_params, _ = self.optimize_hyperparameters(
            X, y, ablation=ablation, groups=groups, use_caching=True
        )

        self._lgbm.set_params(**best_params)
        best_params.setdefault("verbosity", self.verbosity)

        # Fit the model using LightGBM
        self._lgbm.fit(
            X.to_numpy() if isinstance(X, pl.DataFrame) else X,
            y.to_numpy() if isinstance(y, pl.Series) else y,
        )

        return self

    def predict(self, X: MatrixLike) -> VectorLike:
        """Predict using the fitted regressor."""
        return self._lgbm.predict(X)  # type: ignore[return-value]

    def get_lgbm(self) -> LGBMRegressor:
        """Get the underlying LightGBM regressor."""
        if self._lgbm is None:
            raise ValueError("Model has not been fitted yet.")
        return self._lgbm


class LassoEstimator(EstimatorProtocol):
    """Lasso regressor."""

    def __init__(
        self,
        *,
        species: Species,
        group_by: str | None = None,
        cv: int = 5,
        **kwargs: Any,
    ):
        """Initialize the LassoCV regressor."""
        self.species = species
        self.group_by = group_by
        self.cv = cv
        self.lasso_kwargs = kwargs.copy()

        self._model = None

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get the parameters of the regressor."""
        if self._model is None:
            raise ValueError("Model has not been fitted yet.")

        return self._model.get_params(deep=deep)

    def set_params(self, **params: Any) -> LassoEstimator:
        """Set the parameters of the regressor."""
        if self._model is None:
            raise ValueError("Model has not been fitted yet.")

        self._model.set_params(**params)
        return self

    def fit(self, X: MatrixLike, y: VectorLike, **kwargs: Any) -> LassoEstimator:
        """Fit the regressor to the training data."""
        # Extract groups if provided
        groups = kwargs.get("groups", None)

        # Use LassoCV for cross-validating the optimal alpha
        lasso_cv = LassoCV(
            cv=GroupKFold(n_splits=self.cv)
            if self.group_by is not None
            else KFold(n_splits=self.cv),
            verbose=False,
            **self.lasso_kwargs,
        )

        lasso_cv.fit(X, y, groups=to_numpy(groups))
        self._model = Lasso(alpha=lasso_cv.alpha_).fit(X, y)

        return self

    def predict(self, X: MatrixLike) -> VectorLike:
        """Predict using the fitted regressor."""
        if self._model is None:
            raise ValueError("Model has not been fitted yet.")

        return self._model.predict(X)

    def get_sklearn(self) -> Lasso:
        """Get the underlying Lasso regressor."""
        if self._model is None:
            raise ValueError("Model has not been fitted yet.")

        return self._model


@dataclass
class ExperimentResults:
    """Results of an experiment.

    Attributes
    ----------
    species
        Species for which the experiment was run.
    ablation
        Ablation study performed on the model.
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

    X: pl.DataFrame
    metadata: pl.DataFrame
    y_true: pl.Series
    y_pred: Sequence[pl.Series]

    indices: Sequence[dict[Split, np.ndarray]]
    estimators: Sequence[EstimatorProtocol]
    explainers: Sequence[Explainer]

    performances: Sequence[dict[str, float]]

    shap_values: Sequence[Explanation]

    @property
    def num_folds(self) -> int:
        return len(self.y_pred)

    @property
    def features(self) -> list[str]:
        return self.X.columns

    def get_indices(self, fold: int, split: Split) -> np.ndarray:
        """Get indices for the given fold and split."""
        return (
            self.indices[fold][split] if split != "all" else np.arange(self.X.shape[0])
        )

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

        indices = self.get_indices(fold, split)

        return cast(Explanation, shap_values[indices])

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


@dataclass
class CrossValidationResults:
    test_r2: list[float] = field(default_factory=list)
    train_r2: list[float] = field(default_factory=list)
    estimator: list[EstimatorProtocol] = field(default_factory=list)
    indices: dict[Split, list[pl.Series]] = field(
        default_factory=lambda: {"train": [], "test": []}
    )


def train_and_explain(
    species: Species,
    *,
    model_type: ModelType,
    ablation: Ablation = "all",
    group_by: str | None,
    cv: int = 5,
    n_jobs: int = -1,
) -> ExperimentResults:
    """Train models for the given species.

    Parameters
    ----------
    species
        Species to train the model for.
    model_type
        Type of model to use for training, either "gbdt" or "lasso".
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
    X, y = prepare_data(df, ablation)

    # Prepare groups
    if group_by is not None:
        groups = df.select(group_by).to_series()
    else:
        groups = None

    # Cross-validation loop
    print(f"Starting cross-validation for {species} with {model_type} estimator...")

    results = CrossValidationResults()
    splitter = GroupKFold(n_splits=cv) if group_by else KFold(n_splits=cv)
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(to_numpy(X), y, groups=to_numpy(groups))
    ):
        # Create estimator
        if model_type == "gbdt":
            sklearn.set_config(enable_metadata_routing=False)

            estimator = LGBMEstimator(
                species=species,
                group_by=group_by,
                cv=cv,
                n_jobs=n_jobs,
            )
        elif model_type == "lasso":
            # Enable metadata routing for LassoCV to handle group information
            sklearn.set_config(enable_metadata_routing=True)

            estimator = LassoEstimator(
                species=species,
                group_by=group_by,
                cv=cv,
                n_jobs=n_jobs,
                max_iter=25000,
            )

            # Input NaNs are not allowed in LassoCV, so we need to impute them
            X = X.fill_null(0)

            # Standardize the features
            X = pl.DataFrame(
                sklearn.preprocessing.StandardScaler().fit_transform(to_numpy(X)),
                schema=X.schema,
            )
        else:
            raise ValueError(
                f"Unknown estimator: {model_type}. Supported estimators are 'lgbm' and 'lasso'."
            )

        print(f"Fold {fold + 1}/{cv}")

        # Split data into training and test sets
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit the model
        estimator.fit(
            X_train,
            y_train,
            groups=to_numpy(groups[train_idx]) if groups is not None else None,
            ablation=ablation,
        )

        # Evaluate the model
        r2_train = estimator.score(X_train, y_train)
        r2_test = estimator.score(X_test, y_test)

        # Update cross-validation results
        results.test_r2.append(r2_test)
        results.train_r2.append(r2_train)
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

    X_background = X.sample(1000, with_replacement=False)

    for estimator in results.estimator:
        # Create a SHAP explainer for the LGBM model
        if isinstance(estimator, LGBMEstimator):
            explainer = TreeExplainer(
                estimator.get_lgbm(),
                feature_names=X.columns,
                feature_perturbation="tree_path_dependent",
            )
        elif isinstance(estimator, LassoEstimator):
            explainer = LinearExplainer(
                estimator.get_sklearn(),
                feature_names=X.columns,
                masker=IndependentMasker(to_numpy(X_background)),
            )
        else:
            raise ValueError(
                f"Unsupported estimator type: {type(estimator)}. "
                "Supported types are LGBMEstimator and LassoEstimator."
            )

        explainers.append(explainer)
        shap_values.append(explainer(X.to_numpy()))

    return ExperimentResults(
        species=species,
        ablation=ablation,
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
            }
            for fold in range(cv)
        ],
        shap_values=shap_values,
    )
