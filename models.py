# Train models
from config import (
    Species,
    FEATURES,
    SCORING_METRICS,
)
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, GroupKFold, cross_validate
from shap import TreeExplainer, Explanation
import joblib
import optuna
from optuna.trial import Trial

import os
import numpy as np
import polars as pl

from dataclasses import dataclass
from typing import Any, Literal, cast

from data import prepare_data, load_data

Split = Literal["train", "test", "all"]
Estimator = LGBMRegressor


@dataclass
class ExperimentResults:
    """Results of an experiment.

    Attributes
    ----------
    species
        Species for which the experiment was run.
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

    X: pl.DataFrame
    metadata: pl.DataFrame
    y_true: pl.Series
    y_pred: list[pl.Series]

    indices: list[dict[Split, np.ndarray]]
    estimators: list[LGBMRegressor]
    explainers: list[TreeExplainer]

    performances: list[dict[str, float]]

    shap_values: list[Explanation]

    @property
    def num_folds(self) -> int:
        return len(self.y_pred)

    @property
    def features(self) -> list[str]:
        return self.X.columns

    def get_indices(self, fold: int, split: Split) -> np.ndarray | slice:
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
        indices = self.get_indices(fold, split)

        return cast(Explanation, self.shap_values[fold][indices])

    def get_shap_interactions(self, fold: int, split: Split = "test") -> np.ndarray:
        """Get SHAP interaction values for the given fold.

        Parameters
        ----------
        fold
            Fold index.
        split
            Split type ('train', 'test', or 'all').

        Returns
        -------
        SHAP interaction values for the given fold.
        """
        indices = self.get_indices(fold, split)

        interactions = cast(
            np.ndarray,
            self.explainers[fold].shap_interaction_values(self.X[indices].to_numpy()),
        )

        return interactions


def optimize_hyperparameters(
    species: Species,
    cv: int = 5,
    group_col: str | None = "plot_id",
    num_iter: int = 100,
    n_jobs: int = -1,
    use_caching: bool = True,
) -> tuple[dict[str, float], float]:
    """Optimize hyperparameters for a given species.

    Parameters
    ----------
    species
        Species to optimize hyperparameters for.
    cv
        Number of cross-validation folds, by default 5.
    group_col
        Column to group by for cross-validation, by default "plot_id".
    num_iter
        Number of iterations to run, by default 100.
    n_jobs
        Number of jobs to run in parallel, by default -1.
    use_caching
        Whether to use caching for the optimization, by default True.

    Returns
    -------
    A tuple containing the best hyperparameters and the best value found.
    """
    # Check if the study has been cached
    if use_caching and os.path.exists(f"./cache/study-{species}-{group_col}.pkl"):
        study = joblib.load(f"./cache/study-{species}-{group_col}.pkl")
        return study.best_trial.params, study.best_value

    # Load data for the given species
    df = load_data(species)

    # Prepare data
    X, y = prepare_data(df)

    def objective(trial: Trial) -> float:
        # See https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
        grid = {
            # num_leaves is the main parameter to control the complexity of the tree model.
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            # max_depth is also used to control the complexity of the tree model.
            "max_depth": trial.suggest_int("max_depth", -1, 15),
            # min_data_in_leaf is a parameter to prevent over-fitting in a leaf-wise tree.
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 1000),
            # regularization
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_gain_split", 0.0, 1.0),
            # feature sub-sampling and bagging fractiog
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        }

        estimator = LGBMRegressor(**grid, force_row_wise=True, verbose=-1)

        results = cross_validate(
            estimator=estimator,
            X=X,
            y=y,
            groups=df[group_col] if group_col is not None else None,
            scoring=SCORING_METRICS,
            cv=KFold(n_splits=cv) if group_col is None else GroupKFold(n_splits=cv),
            n_jobs=n_jobs,
            verbose=False,
        )

        return results["test_r2"].mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=num_iter)

    print(f"Best parameters found: {study.best_params}")
    print(f"with test R2: {study.best_value}")

    if use_caching:
        joblib.dump(study, f"./cache/study-{species}-{group_col}.pkl")

    return study.best_trial.params, study.best_value


def train_and_explain(
    species: Species,
    params: dict[str, Any],
    cv: int = 5,
    group_col: str | None = "plot_id",
    n_jobs: int = -1,
    verbosity: int = 0,
) -> ExperimentResults:
    """Train models for the given species.

    Parameters
    ----------
    species
        Species to train the model for.
    cv
        Number of cross-validation folds, by default 5.
    group_col
        Column to group by for cross-validation, by default "plot_id".
    n_jobs
        Number of jobs to run in parallel, by default -1.
    verbosity
        Verbosity level of LightGBM, by default 0.

    Returns
    -------
    A `Result` object containing the trained models and SHAP values.
    """
    print(f"Training model for {species}")

    # Load data for the given species
    df = load_data(species)

    # Prepare data
    X, y = prepare_data(df)

    # Set verbosity level
    params = {**params, "verbosity": verbosity}

    # Train cv models
    results = cross_validate(
        estimator=LGBMRegressor(**params, force_row_wise=True),
        X=X,
        y=y,
        groups=df[group_col] if group_col is not None else None,
        scoring=SCORING_METRICS,
        cv=KFold(n_splits=cv) if group_col is None else GroupKFold(n_splits=cv),
        n_jobs=n_jobs,
        return_estimator=True,
        return_train_score=True,
        return_indices=True,
    )

    print(f"Finished training model for {species}")
    print("Performance:")
    print(
        f" `- R2 (test): {results['test_r2'].mean():.2f} +/- {results['test_r2'].std():.2f}"
    )
    print(
        f" `- R2 (train): {results['train_r2'].mean():.2f} +/- {results['train_r2'].std():.2f}"
    )

    # Explain model
    explainers = []
    shap_values = []

    for estimator in results["estimator"]:
        explainer = TreeExplainer(
            estimator,
            feature_names=FEATURES,
            feature_perturbation="tree_path_dependent",
        )

        explainers.append(explainer)
        shap_values.append(explainer(X.to_numpy()))

    return ExperimentResults(
        species=species,
        X=X,
        metadata=df.select(pl.selectors.exclude(*FEATURES)),
        y_true=y,
        y_pred=[
            pl.Series("y_pred", model.predict(X)) for model in results["estimator"]
        ],
        indices=[
            {
                "train": results["indices"]["train"][fold],
                "test": results["indices"]["test"][fold],
            }
            for fold in range(cv)
        ],
        estimators=results["estimator"],
        explainers=explainers,
        performances=[
            {
                "test_r2": float(results["test_r2"][fold]),
                "train_r2": float(results["train_r2"][fold]),
            }
            for fold in range(cv)
        ],
        shap_values=shap_values,
    )
