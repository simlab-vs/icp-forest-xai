# Train models
from common import (
    Species,
    LIGHTGBM_PARAMS,
    FEATURES,
    TARGET,
    CATEGORICAL_COLUMNS,
    SCORING_METRICS,
)
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold, cross_validate
from shap import TreeExplainer as TreeExplainer

import numpy as np
import polars as pl

from dataclasses import dataclass
from typing import Literal

from data import cat_to_codes, load_data

Split = Literal["train", "test"]


@dataclass
class Result:
    X: pl.DataFrame
    y_true: pl.Series
    y_pred: list[pl.Series]

    indices: list[dict[Split, np.ndarray]]
    estimators: list[LGBMRegressor]
    explainers: list[TreeExplainer]

    performances: list[dict[str, float]]

    shap_values: list[np.ndarray]

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
        return (
            self.X[self.indices[fold][split]],
            self.y_true[self.indices[fold][split]],
            self.y_pred[fold][self.indices[fold][split]],
        )

    def get_shap_values(self, fold: int, split: Split) -> np.ndarray:
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
        return self.shap_values[fold][self.indices[fold][split]]

    def get_shap_interactions(self, fold: int, split: Split) -> np.ndarray:
        """Get SHAP interaction values for the given fold.

        Parameters
        ----------
        fold
            Fold index.

        Returns
        -------
        SHAP interaction values for the given fold.
        """
        return self.explainers[fold].shap_interaction_values(
            self.X[self.indices[fold][split]].to_numpy()
        )


def train_and_explain(
    species: Species,
    cv: int = 5,
    group_col: str = "tree_id",
    n_jobs: int = -1,
) -> Result:
    """Train models for the given species.

    Parameters
    ----------
    species
        Species to train the model for.
    cv
        Number of cross-validation folds, by default 5.
    group_col
        Column to group by for cross-validation, by default "tree_id".
    n_jobs
        Number of jobs to run in parallel, by default -1.
    n_samples_interactions
        Number of samples to use for SHAP interaction values, by default 1000.

    Returns
    -------
    A `Result` object containing the trained models and SHAP values.
    """
    print(f"Training model for {species}")

    # Load data for the given species
    df = load_data(species)

    # Prepare data
    X = cat_to_codes(df[FEATURES], CATEGORICAL_COLUMNS)
    y = df[TARGET]

    # TODO: perform hyperparameter tuning
    estimator = LGBMRegressor(**LIGHTGBM_PARAMS[species], force_row_wise=True)

    results = cross_validate(
        estimator=estimator,
        X=X,
        y=y,
        # TODO: investigate effect of grouping by plot_id
        groups=df[group_col],
        scoring=SCORING_METRICS,
        cv=GroupKFold(n_splits=cv, shuffle=True),
        n_jobs=n_jobs,
        return_estimator=True,
        return_indices=True,
    )

    print(f"Finished training model for {species}")
    print("Performance:")
    print(f"`- R2: {results['test_r2'].mean():.2f} +/- {results['test_r2'].std():.2f}")
    print(
        f"`- MAE: {results['test_neg_mean_absolute_error'].mean() * 100:.2f}% +/- {results['test_neg_mean_absolute_error'].std() * 100:.2f}%"
    )

    # Explain model
    explainers = []
    shap_values = []
    performances = []
    for model in results["estimator"]:
        performances.append(
            {
                "r2": results["test_r2"],
                "mae": results["test_neg_mean_absolute_error"],
            }
        )

        explainer = TreeExplainer(
            model, feature_names=FEATURES, feature_perturbation="tree_path_dependent"
        )

        explainers.append(explainer)
        shap_values.append(explainer(X.to_numpy()))

    return Result(
        X=X,
        y_true=y,
        y_pred=[model.predict(X) for model in results["estimator"]],
        indices=[
            {
                "train": results["indices"]["train"][fold],
                "test": results["indices"]["test"][fold],
            }
            for fold in range(cv)
        ],
        estimators=results["estimator"],
        explainers=explainers,
        performances=performances,
        shap_values=shap_values,
    )


if __name__ == "__main__":
    models = train_and_explain("spruce")
    print(models)
