import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from typing import cast

from models import Estimator, ExperimentResults, Split

import os


def plot_ceteris_paribus_profile(
    estimator: Estimator,
    X: pl.DataFrame,
    instance_id: int,
    feature: str,
    ax: Axes | None = None,
):
    """
    Plot ceteris paribus profile for a given feature.

    Parameters
    ----------
    estimator
        Fitted model.
    X
        Dataframe containing the features.
    instance_id
        Index of the instance for which to plot the profile
    feature
        Name of the feature for which to plot the profile.
    ax
        Axes object to plot the profile on. If None, a new figure is created.

    Returns
    -------
    Tuple containing the feature range and the corresponding predictions.
    """
    # A few checks up front
    if instance_id >= X.shape[0] or instance_id < 0:
        raise ValueError("Instance index out of bounds")
    if feature not in X.columns:
        raise ValueError("Feature not found in the dataframe")

    # Get the corresponding row
    instance = X.slice(instance_id, length=1)

    # Get the range of the feature
    feature_range = np.linspace(
        X.select(pl.col(feature).drop_nans().min()).item(),
        X.select(pl.col(feature).drop_nans().max()).item(),
        num=100,
    )

    X_cp = pl.concat(
        [instance.with_columns(pl.lit(f).alias(feature)) for f in feature_range],
        how="vertical",
    )

    y_pred = estimator.predict(X_cp)

    if ax is None:
        plt.figure(figsize=(6, 4))
        ax = plt.gca()

    ax.plot(feature_range, y_pred)  # type: ignore

    # Draw circle at the instance value
    ax.scatter(
        instance.item(0, feature),
        estimator.predict(instance),  # type: ignore
        color="red",
        s=100,
        label="Instance",
        alpha=1.0,
    )

    ax.set_xlabel(feature)
    ax.set_ylabel("Predicted value")

    return feature_range, y_pred


def plot_interaction_matrix(
    results: ExperimentResults,
    *,
    num_samples: int | None = 2000,
    fold: int = 0,
    split: Split = "all",
    top_n: int | list[str] = 20,
    vmax: float | None = None,
    ax: Axes | None = None,
    use_caching: bool = True,
    disable_plotting: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Plot SHAP interaction matrix for the given fold.

    Parameters
    ----------
    results
        Results object containing the SHAP values.
    num_samples
        Number of samples to use for the SHAP interaction values (None for all samples).
    fold
        Fold index to be used (by default, the first fold).
    top_n
        Number of features or list of features to include in the plot.
    vmax
        Maximum value for the color scale (None for automatic scaling).
    ax
        Axes object to plot the matrix on. If None, a new figure is created.
    use_caching
        Whether to use caching for the interaction values.
    disable_plotting
        Whether to plot the matrix (set to only get the results).

    Returns
    -------
    A tuple (interactions, indices) containing the interaction values and the indices of the
    features.
    """

    def _get_fold_data(fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Load from cache
        if use_caching:
            try:
                cached = pl.read_parquet(
                    os.path.join(
                        "cache", f"interactions-{results.species}-{fold}.parquet"
                    )
                )

                return (
                    cached["interactions"].to_numpy(),
                    cached["shap_values"].to_numpy(),
                    cached["indices"].to_numpy(),
                )
            except FileNotFoundError:
                pass

        # Get the SHAP interaction values
        interactions, indices = results.get_shap_interactions(fold, split, num_samples)

        # Get the SHAP values for the feature
        shap_values = cast(
            np.ndarray, results.get_shap_values(fold, split).values[indices]
        )

        assert shap_values.shape[0] == interactions.shape[0]

        # Cache the values
        if use_caching:
            pl.DataFrame(
                {
                    "interactions": interactions,
                    "shap_values": shap_values,
                    "indices": indices,
                }
            ).write_parquet(
                os.path.join("cache", f"interactions-{results.species}-{fold}.parquet")
            )

        return interactions, shap_values, indices

    # Get the data for the fold
    interactions, shap_values, indices = _get_fold_data(fold)

    # Ensure that interaction values sum up to SHAP values
    # This is an important consistency check, especially when loading cached values that
    # might have been computed for different conditions.
    assert np.all(np.abs(shap_values - np.sum(interactions, axis=2)) < 1e-9)

    if not disable_plotting:
        # Get the top-n features with the highest interaction values
        if isinstance(top_n, int):
            top_n_idx = np.argsort(np.absolute(shap_values).mean(axis=0))[::-1][:top_n]
        else:
            top_n_idx = [results.X.columns.index(f) for f in top_n]
            top_n = len(top_n_idx)

        interacts_no_diag = np.vectorize(
            lambda m: m - np.diag(np.diag(m)), signature="(m, m)->(m, m)"
        )(interactions)

        interacts_no_diag = interacts_no_diag[:, top_n_idx, :][:, :, top_n_idx]

        if ax is None:
            plt.figure(figsize=(10, 8))
            ax = plt.gca()

        pcm = ax.imshow(
            np.absolute(interacts_no_diag).mean(axis=0),
            cmap="coolwarm",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_xticks(np.arange(top_n), [results.X.columns[idx] for idx in top_n_idx])
        ax.set_yticks(np.arange(top_n), [results.X.columns[idx] for idx in top_n_idx])
        ax.tick_params(axis="x", rotation=90)

        cbar = plt.colorbar(
            pcm, ax=ax, label="Mean interaction value", shrink=0.8, pad=0.02
        )
        cbar.set_label("Mean absolute interaction value")

    return interactions, indices
