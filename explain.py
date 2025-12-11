import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import seaborn as sns
import os

from enum import Enum
from typing import Callable, cast, Any

from models import EstimatorProtocol, ExperimentResults, Split


class PlotType(Enum):
    SCATTER = "scatter"
    LINE = "line"
    DENSITY = "density"


def plot_dependence(
    results: ExperimentResults,
    feature: str,
    fold: int | None = None,
    label: str | None = None,
    show_no_effect: bool = True,
    fit_func: Callable | None = None,
    fit_p0: tuple[float, float, float] | None = None,
    fit_formula: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ax: Axes | None = None,
    color: str = "#1f77b4",
    plot_type: PlotType = PlotType.SCATTER,
    **kwargs: Any,
) -> Axes:
    """Plot SHAP dependence plot for a given feature.

    Parameters
    ----------
    results
        Results object containing the SHAP values.
    feature
        Name of the feature for which to plot the SHAP values.
    fold
        Fold index to be used (by default, all folds).
    label
        Label for the plot. If None, no label is set.
    show_no_effect
        Whether to show the line indicating no effect (default is True).
    fit_func
        Function to fit a curve to the data (default is None).
    fit_p0
        Initial parameters for the fit function (default is None).
    fit_formula
        The formula to display the fitted curve (default is None).
    xlim
        Tuple specifying the x-axis limits. If None, limits are set based on the data.
    ylim
        Tuple specifying the y-axis limits. If None, limits are set based on the data.
    ax
        Axes object to plot the SHAP values on. If None, a new figure is created.
    color
        Color of the scatter points.
    plot_type
        Type of plot to create (scatter or line).
    **kwargs
        Additional keyword arguments to pass to the scatter plot.

    Returns
    -------
    The axes object with the SHAP dependence plot.
    """
    # If no alpha is provided, set it to 0.6
    kwargs.setdefault("alpha", 0.6)

    if fold is None:
        indices = np.arange(results.X.shape[0])
        shap_values = (
            np.concatenate(
                [
                    results.shap_values[fold][:, feature].values  # type: ignore
                    for fold in range(results.num_folds)
                ],
                dtype=np.float64,
            )
        ) * 100
        feature_values = np.concatenate(
            [
                results.shap_values[fold][:, feature].data  # type: ignore
                for fold in range(results.num_folds)
            ],
            dtype=np.float64,
        )
    else:
        shap_struct = results.shap_values[fold][:, feature]
        assert shap_struct is not None, (
            f"Feature '{feature}' not found in SHAP values for fold {fold}"
        )

        indices = results.get_indices(fold, "all")

        shap_values = (cast(np.ndarray, shap_struct.values).astype(np.float64)) * 100  # type: ignore
        feature_values = cast(np.ndarray, shap_struct.data).astype(np.float64)  # type: ignore

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    # Enlarge slightly the limits for better visibility
    if xlim is None:
        xlim = (
            np.nanmin(results.X[indices, feature]),
            np.nanmax(results.X[indices, feature]),
        )

    if ylim is None:
        ylim = (
            np.nanmin(shap_values),
            np.nanmax(shap_values),
        )  # type: ignore

    xlim = (xlim[0] - 0.05 * (xlim[1] - xlim[0]), xlim[1] + 0.05 * (xlim[1] - xlim[0]))
    ylim = (ylim[0] - 0.05 * (ylim[1] - ylim[0]), ylim[1] + 0.05 * (ylim[1] - ylim[0]))  # type: ignore

    # Get indices for which the feature values are not NaN
    valid_indices = ~np.isnan(feature_values)
    xwidth = xlim[1] - xlim[0]
    ywidth = ylim[1] - ylim[0]

    if plot_type == PlotType.LINE:
        sns.lineplot(
            x=feature_values[valid_indices],
            y=shap_values[valid_indices],
            ax=ax,
            color=color,
            label=label,
            errorbar=("pi", 95),
            **kwargs,
        )
    elif plot_type == PlotType.SCATTER:
        wiggle = 0.005
        xwiggle = np.random.uniform(
            -wiggle * xwidth, wiggle * xwidth, size=np.sum(valid_indices)
        )
        ywiggle = np.random.uniform(
            -wiggle * ywidth, wiggle * ywidth, size=np.sum(valid_indices)
        )
        sns.scatterplot(
            x=feature_values[valid_indices] + xwiggle,
            y=shap_values[valid_indices] + ywiggle,
            ax=ax,
            color=color,
            edgecolor=None,
            legend=False,
            size=6,
            label="_nolegend_",
            **kwargs,
        )

        if fit_func is not None:
            # Fit a power-law curve to the data
            from scipy.optimize import curve_fit

            popt, _ = curve_fit(
                fit_func,
                feature_values[valid_indices],
                shap_values[valid_indices],
                p0=(1.0, 1.0, 0.0) if fit_p0 is None else fit_p0,
            )

            x_fit = np.linspace(xlim[0], xlim[1], 100)
            y_fit = fit_func(x_fit, *popt)

            if fit_formula is not None:
                label = f"${fit_formula.format(*popt)}$"
            else:
                label = "Fitted curve"

            ax.plot(x_fit, y_fit, color="k", linestyle="--", linewidth=2, label=label)

            if label is not None:
                ax.legend([label])
    elif plot_type == PlotType.DENSITY:
        # Overlaid inset axes for histogram with the same x-axis limits
        ax2 = ax.inset_axes(
            bounds=(0, 0, 1.0, 0.2),
            zorder=0,
            sharex=ax,
            frame_on=False,
        )

        # Remove xticks/yticks from the inset axes
        ax2.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        ax2.tick_params(
            axis="y",
            which="both",
            left=False,
            right=False,
            labelleft=False,
            labelright=False,
        )

        # Overlaid histogram of point density
        sns.histplot(
            x=feature_values[valid_indices],
            legend=False,
            ax=ax2,
            bins=50,
            stat="density",
            color="grey",
            alpha=0.3,
            edgecolor=None,
        )

        if label is not None:
            ax.collections[-1].set_label(label)
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")

    # Draw the line that indicates no effect
    if show_no_effect:
        ax.axhline(0, color="grey", linestyle="--")
        ax.text(
            xlim[1] - 0.02 * xwidth,
            0.02 * ywidth,
            "No effect",
            color="grey",
            ha="right",
        )

    # Set vertical grid lines for better readability
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)

    ax.set_title(results.species.capitalize())
    ax.set_xlabel(feature)
    ax.set_ylabel("SHAP value [%]")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    fig = ax.get_figure()

    if fig is not None and isinstance(fig, Figure):
        fig.tight_layout()

    return ax


def plot_ceteris_paribus_profile(
    estimator: EstimatorProtocol,
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


def compute_interaction_matrix(
    results: ExperimentResults,
    *,
    num_samples: int | None = 2000,
    fold: int = 0,
    split: Split = "all",
    top_n: int | list[str] = 20,
    vmax: float | None = None,
    ax: Axes | None = None,
    use_caching: bool = True,
    plotting: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute SHAP interaction matrix for the given fold and plot it optionally.

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
    plotting
        Plot the interaction matrix if True (or if `ax` is not None).

    Returns
    -------
    A tuple (interactions, indices) containing the interaction values and the indices of the
    features.
    """
    import cmocean

    def _get_fold_data(fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        fname = os.path.join(
            "cache", f"interactions-{results.species}-{results.ablation}-{fold}.parquet"
        )

        # Load from cache
        if use_caching:
            try:
                cached = pl.read_parquet(fname)

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
            ).write_parquet(fname)

        return interactions, shap_values, indices

    # Get the data for the fold
    interactions, shap_values, indices = _get_fold_data(fold)

    # Ensure that interaction values sum up to SHAP values
    # This is an important consistency check, especially when loading cached values that
    # might have been computed for different conditions.
    assert np.all(np.abs(shap_values - np.sum(interactions, axis=2)) < 1e-9)

    if plotting or ax is not None:
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
            cmap=cmocean.cm.thermal,  # type: ignore[attr-defined]
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
