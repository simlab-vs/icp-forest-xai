import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import seaborn as sns
import os

from enum import Enum
from typing import Callable, Sequence, cast, Any

from scipy import stats as _scipy_stats

from models import EstimatorProtocol, ExperimentResults, Split
from config import FEATURES_METADATA


def _feature_label(feature: str) -> str:
    meta = FEATURES_METADATA.get(feature, {})
    label: str = meta.get("label") or feature
    unit: str | None = meta.get("unit")
    return f"{label} [{unit}]" if unit else label


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
    use_percentage: bool = True,
    fit_func: Callable | None = None,
    fit_p0: tuple[float, float, float] | None = None,
    fit_formula: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ax: Axes | None = None,
    color: str = "#1f77b4",
    plot_type: PlotType = PlotType.SCATTER,
    with_density: bool = True,
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
    use_percentage
        Whether to express SHAP values as percentile rank percentages.
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
    with_density
        Whether to overlay a density histogram at the bottom of the plot.
    **kwargs
        Additional keyword arguments to pass to the scatter plot.

    Returns
    -------
    The axes object with the SHAP dependence plot.
    """
    # If no alpha is provided, set it to 0.6
    kwargs.setdefault("alpha", 0.6)

    y_label = "SHAP value [%]" if use_percentage else "SHAP value"

    if fold is None:
        indices = np.arange(results.X.shape[0])
        shap_values = np.concatenate(
            [
                results.shap_values[f][:, feature].values
                for f in range(results.num_folds)
            ],
            dtype=np.float64,
        )
        feature_values = np.concatenate(
            [results.shap_values[f][:, feature].data for f in range(results.num_folds)],
            dtype=np.float64,
        )
    else:
        shap_struct = results.shap_values[fold][:, feature]
        assert shap_struct is not None, (
            f"Feature '{feature}' not found in SHAP values for fold {fold}"
        )
        indices = results.get_indices(fold, "all")
        shap_values = cast(np.ndarray, shap_struct.values).astype(np.float64)
        feature_values = cast(np.ndarray, shap_struct.data).astype(np.float64)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    if use_percentage:
        shap_values = shap_values * 100

    # Define x and y limits if not provided
    if xlim is None:
        xlim = (
            np.nanmin(results.X[indices, feature]),
            np.nanmax(results.X[indices, feature]),
        )

    if ylim is None:
        ylim = (
            np.nanmin(shap_values),
            np.nanmax(shap_values),
        )

    # Filter out NaN values and out of bounds values
    valid_indices = (
        ~np.isnan(feature_values)
        & (xlim[0] <= feature_values)
        & (feature_values <= xlim[1])
    )

    # Add some padding to the limits
    xlim = (xlim[0] - 0.05 * (xlim[1] - xlim[0]), xlim[1] + 0.05 * (xlim[1] - xlim[0]))
    ylim = (ylim[0] - 0.05 * (ylim[1] - ylim[0]), ylim[1] + 0.05 * (ylim[1] - ylim[0]))

    xwidth = xlim[1] - xlim[0]
    ywidth = ylim[1] - ylim[0]

    # Convert to percentage for better interpretability
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
            -wiggle * xwidth,
            wiggle * xwidth,
            size=np.sum(valid_indices),
        )
        ywiggle = np.random.uniform(
            -wiggle * ywidth,
            wiggle * ywidth,
            size=np.sum(valid_indices),
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

    if plot_type == PlotType.DENSITY or with_density:
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
            binrange=xlim,
            stat="density",
            color="grey",
            alpha=0.3,
            edgecolor=None,
        )

        if label is not None:
            ax.collections[-1].set_label(label)

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
    ax.set_xlabel(_feature_label(feature))
    ax.set_ylabel(y_label)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    fig = ax.get_figure()

    if fig is not None and isinstance(fig, Figure):
        fig.tight_layout()

    return ax


def _compute_cp_matrix(
    rows: pl.DataFrame,
    feature: str,
    grid: np.ndarray,
    estimators: Sequence[EstimatorProtocol],
) -> np.ndarray:
    """Return shape (n_grid, n_rows) predictions averaged across estimators.

    For each grid value the feature column is overwritten while all other
    columns stay fixed (ceteris paribus), then predictions are averaged over
    the estimator ensemble.
    """
    n_grid = len(grid)
    out = np.empty((n_grid, len(rows)))
    for gi, v in enumerate(grid):
        X_mod = rows.with_columns(pl.lit(float(v)).alias(feature))
        preds_all = np.stack(
            [np.asarray(est.predict(X_mod), dtype=np.float64) for est in estimators]
        )
        out[gi] = preds_all.mean(axis=0)
    return out


def plot_partial_dependence_orig_space(
    results: ExperimentResults,
    features: list[str],
    fold: int = 0,
    n_grid: int = 50,
    n_samples: int = 500,
    axes: np.ndarray | None = None,
    freq_bins: int = 20,
) -> tuple[Figure, np.ndarray]:
    """Partial dependence plots in original growth-rate units (%/yr).

    Parameters
    ----------
    results
        Fitted ExperimentResults with dist_params set.
    features
        Feature names to plot, one panel each.
    fold
        Fold index; training split of this fold is used as background.
    n_grid
        Grid resolution (5th-95th percentile of training values).
    n_samples
        Number of background samples drawn from the training split.
    axes
        Pre-created array of Axes; created automatically if None.
    freq_bins
        Bins for the secondary frequency histogram.

    Returns
    -------
    fig, axes
    """
    if results.dist_params is None:
        raise ValueError("dist_params is None; cannot compute PD in original space.")

    dist_params = results.dist_params

    n_panels = len(features)
    ncols = min(n_panels, 2)
    nrows = (n_panels + 1) // 2

    if axes is None:
        fig, axes_arr = plt.subplots(
            nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False
        )
    else:
        axes_arr = axes.reshape(nrows, ncols)
        fig_obj = axes_arr.flat[0].get_figure()
        assert fig_obj is not None
        fig = fig_obj

    X_train, _, _ = results.get_data(fold, "train")
    rng = np.random.default_rng(0)
    bg_idx = rng.choice(len(X_train), size=min(n_samples, len(X_train)), replace=False)
    bg = X_train[bg_idx]

    os.makedirs("./figures", exist_ok=True)

    for panel_idx, feature in enumerate(features):
        ax = axes_arr.flat[panel_idx]

        feat_vals = X_train[feature].drop_nulls().to_numpy()
        grid = np.linspace(
            np.percentile(feat_vals, 5), np.percentile(feat_vals, 95), n_grid
        )

        preds_grid = _compute_cp_matrix(bg, feature, grid, results.estimators)

        flat = pl.Series(preds_grid.ravel())
        y_orig_flat = results.get_inverse_transform(flat, dist_params).to_numpy() * 100
        y_orig = y_orig_flat.reshape(n_grid, len(bg))

        pd_mean = y_orig.mean(axis=1)

        center = pd_mean[n_grid // 2]
        pd_mean = pd_mean - center
        y_orig_centered = y_orig - y_orig[n_grid // 2, :]

        xlim = (float(grid[0]), float(grid[-1]))
        pad = 0.05 * (xlim[1] - xlim[0])
        xlim_padded = (xlim[0] - pad, xlim[1] + pad)

        # Compute binned SHAP mean/std before plotting so we can derive PD y-limits
        shap_exp = results.get_shap_values(fold, "all")
        shap_mean = np.full(n_grid, np.nan)
        shap_std = np.full(n_grid, np.nan)
        if feature in results.features:
            feat_idx = results.features.index(feature)
            shap_vals = shap_exp.values[:, feat_idx]
            shap_feat = shap_exp.data[:, feat_idx]
            bin_edges = np.concatenate(
                [[grid[0]], (grid[1:] + grid[:-1]) / 2, [grid[-1]]]
            )
            for gi in range(n_grid):
                mask = (shap_feat >= bin_edges[gi]) & (shap_feat < bin_edges[gi + 1])
                if mask.sum() > 1:
                    shap_mean[gi] = shap_vals[mask].mean()
                    shap_std[gi] = shap_vals[mask].std()

        # Find PD y-axis limits that minimise apparent MAD vs SHAP mean in display space.
        # Display coord of pd:   (pd - y_lo) / (y_hi - y_lo)
        # Display coord of shap: (shap - shap_lo) / shap_range
        # We fit shap_norm = a * pd_mean + b via Theil-Sen (L1-ish), then
        # invert to get y_lo = -b/a and y_hi = (1-b)/a.
        valid = ~np.isnan(shap_mean)
        y_lo_pd: float | None = None
        y_hi_pd: float | None = None
        n_clip_hi = n_clip_lo = 0
        ext_hi = ext_lo = 0.0
        if valid.sum() > 2:
            from scipy.stats import theilslopes

            shap_lo_ax = float(np.nanmin(shap_mean - shap_std))
            shap_hi_ax = float(np.nanmax(shap_mean + shap_std))
            shap_range = shap_hi_ax - shap_lo_ax or 1.0
            s_norm = (shap_mean[valid] - shap_lo_ax) / shap_range
            fit = theilslopes(s_norm, pd_mean[valid])
            a, b = float(fit.slope), float(fit.intercept)
            if abs(a) > 1e-10:
                y_lo_pd = -b / a
                y_hi_pd = (1.0 - b) / a
                if y_lo_pd > y_hi_pd:
                    y_lo_pd, y_hi_pd = y_hi_pd, y_lo_pd

                traj_max = y_orig_centered.max(axis=0)
                traj_min = y_orig_centered.min(axis=0)
                n_clip_hi = int((traj_max > y_hi_pd).sum())
                n_clip_lo = int((traj_min < y_lo_pd).sum())
                ext_hi = float(traj_max.max() - y_hi_pd) if n_clip_hi > 0 else 0.0
                ext_lo = float(y_lo_pd - traj_min.min()) if n_clip_lo > 0 else 0.0

        for i in range(y_orig_centered.shape[1]):
            ax.plot(
                grid, y_orig_centered[:, i], color="#1f77b4", linewidth=0.5, alpha=0.05
            )
        ax.plot(grid, pd_mean, color="#1f77b4", linewidth=2)
        ax.axhline(0, color="grey", linestyle="--")

        if y_lo_pd is not None and y_hi_pd is not None:
            ax.set_ylim(y_lo_pd, y_hi_pd)

        ax.xaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlabel(_feature_label(feature))
        ax.set_ylabel("Partial dependence [%/yr]", color="#1f77b4")
        ax.tick_params(axis="y", labelcolor="#1f77b4")
        ax.set_title(results.species.capitalize())
        ax.set_xlim(xlim_padded)

        if n_clip_hi > 0:
            ax.annotate(
                f"{n_clip_hi} clipped\n+{ext_hi:.2f}%/yr max",
                xy=(0.5, 0.99),
                xytext=(0.5, 0.84),
                xycoords="axes fraction",
                textcoords="axes fraction",
                fontsize=7,
                ha="center",
                va="top",
                color="#1f77b4",
                arrowprops=dict(arrowstyle="->", color="#1f77b4"),
            )
        if n_clip_lo > 0:
            ax.annotate(
                f"{n_clip_lo} clipped\n−{ext_lo:.2f}%/yr max",
                xy=(0.5, 0.01),
                xytext=(0.5, 0.16),
                xycoords="axes fraction",
                textcoords="axes fraction",
                fontsize=7,
                ha="center",
                va="bottom",
                color="#1f77b4",
                arrowprops=dict(arrowstyle="->", color="#1f77b4"),
            )

        # SHAP mean ± std on a second y-axis
        shap_color = "#d62728"
        ax_shap = ax.twinx()
        ax_shap.plot(
            grid, shap_mean * 100, color=shap_color, linewidth=1.5, linestyle="--"
        )
        ax_shap.fill_between(
            grid,
            shap_mean * 100 - shap_std * 100,
            shap_mean * 100 + shap_std * 100,
            color=shap_color,
            alpha=0.15,
        )
        ax_shap.axhline(0, color=shap_color, linestyle=":", linewidth=0.8, alpha=0.5)
        ax_shap.set_ylabel("SHAP value [percentile rank %]", color=shap_color)
        ax_shap.tick_params(axis="y", labelcolor=shap_color)

        ax2 = ax.inset_axes(
            bounds=(0, 0, 1.0, 0.2), zorder=0, sharex=ax, frame_on=False
        )
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
        train_feat_vals = X_train[feature].drop_nulls().to_numpy()
        sns.histplot(
            x=train_feat_vals,
            legend=False,
            ax=ax2,
            bins=freq_bins,
            binrange=xlim_padded,
            stat="density",
            color="grey",
            alpha=0.3,
            edgecolor=None,
        )

        fig.savefig(
            f"./figures/partial_dependence_orig_space_{results.species}_{feature}.pdf",
            bbox_inches="tight",
        )

    for j in range(panel_idx + 1, nrows * ncols):
        axes_arr.flat[j].set_visible(False)

    fig.tight_layout()
    return fig, axes_arr


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

    y_pred = _compute_cp_matrix(instance, feature, feature_range, [estimator]).ravel()

    if ax is None:
        plt.figure(figsize=(6, 4))
        ax = plt.gca()

    ax.plot(feature_range, y_pred)

    # Draw circle at the instance value
    ax.scatter(
        instance.item(0, feature),
        estimator.predict(instance),
        color="red",
        s=100,
        label="Instance",
        alpha=1.0,
    )

    ax.set_xlabel(_feature_label(feature))
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
            cmap=cmocean.cm.thermal,  # type: ignore
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


# ---------------------------------------------------------------------------
# SHAP dependence curve stability across CV blocking strategies
# ---------------------------------------------------------------------------

_STRATEGY_COLORS = {
    "standard": "#1f77b4",
    "temporal": "#ff7f0e",
    "spatial": "#2ca02c",
}


def _get_shap_arrays(
    results: "Any",  # ExperimentResults — avoid circular import in type hint
    feature: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate SHAP values and feature values for `feature` across all folds."""
    shap_vals = np.concatenate(
        [
            results.shap_values[fold][:, feature].values
            for fold in range(results.num_folds)
        ]
    ).astype(np.float64)
    feat_vals = np.concatenate(
        [
            results.shap_values[fold][:, feature].data
            for fold in range(results.num_folds)
        ]
    ).astype(np.float64)
    return shap_vals, feat_vals


def _quantile_edges(values: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Compute n_bins+1 quantile-based bin edges from non-NaN values."""
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)
    return np.nanpercentile(valid, np.linspace(0.0, 100.0, n_bins + 1))


def _binned_shap_mean_counts(
    feat_vals: np.ndarray,
    shap_vals: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-bin mean SHAP and observation counts.

    Returns arrays of length ``len(edges) - 1``; bins with no data are NaN / 0.
    """
    n_bins = len(edges) - 1
    bin_means = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins)

    valid = ~np.isnan(feat_vals) & ~np.isnan(shap_vals)
    if not valid.any():
        return bin_means, bin_counts

    fv, sv = feat_vals[valid], shap_vals[valid]
    ids = np.searchsorted(edges[1:-1], fv, side="right")

    for b in range(n_bins):
        m = ids == b
        if m.sum():
            bin_means[b] = sv[m].mean()
            bin_counts[b] = float(m.sum())

    return bin_means, bin_counts


def _binned_fold_envelope(
    results: "Any",
    feature: str,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean ± std of per-fold bin means and pooled observation counts.

    The band is mean ± 1 std across folds, so it is always symmetric around
    the reported mean line.  Using Q25/Q75 would NOT guarantee the mean falls
    inside the band when the distribution of fold estimates is skewed.

    Returns (bin_means, bin_lo, bin_hi, bin_counts).
    """
    n_bins = len(edges) - 1
    fold_bin_means = []
    pooled_counts = np.zeros(n_bins)

    for fold in range(results.num_folds):
        sv = results.shap_values[fold][:, feature].values.astype(np.float64)
        fv = results.shap_values[fold][:, feature].data.astype(np.float64)
        fm, fc = _binned_shap_mean_counts(fv, sv, edges)
        fold_bin_means.append(fm)
        pooled_counts += fc

    fold_arr = np.stack(fold_bin_means, axis=0)  # shape: [n_folds, n_bins]
    with np.errstate(all="ignore"), np.testing.suppress_warnings() as _sw:
        _sw.filter(RuntimeWarning)
        bin_means = np.nanmean(fold_arr, axis=0)
        bin_std = np.nanstd(fold_arr, axis=0, ddof=0)

    # NaN where all folds had no data in that bin
    no_data = np.all(np.isnan(fold_arr), axis=0)
    bin_means[no_data] = np.nan
    bin_std[no_data] = np.nan

    bin_lo = bin_means - bin_std
    bin_hi = bin_means + bin_std

    return bin_means, bin_lo, bin_hi, pooled_counts


def _plot_shap_stability_figure(
    species: str,
    feature_raw: str,
    feature_label: str,
    panels: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]],
    bin_centers: np.ndarray,
    bin_widths: np.ndarray,
    obs_counts: np.ndarray,
    figures_dir: str,
    stability_metrics: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Save a multi-panel SHAP dependence stability figure.

    Parameters
    ----------
    feature_raw
        Raw feature column name — used for the output filename.
    feature_label
        Human-readable feature name with unit, e.g. ``"Mean defoliation [%]"``
        — used for axis labels and the figure title.
    panels
        Each entry is ``(strategy_title, color, fp_means, fp_q25, fp_q75)``.
    stability_metrics
        Maps strategy_title → (spearman_rho, normalized_mad) for non-baseline
        panels.  Absent keys (e.g. the standard panel) get no metric annotation.
    obs_counts
        Per-bin observation counts from the full (unfolded) dataset — used for
        the frequency histogram inset.
    """
    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5.5), squeeze=False)

    # Shared y limits derived from all Q25/Q75 bands
    all_band_vals: list[np.ndarray] = []
    for _, _, _, q25, q75 in panels:
        all_band_vals.extend([q25[~np.isnan(q25)], q75[~np.isnan(q75)]])
    if any(v.size for v in all_band_vals):
        combined = np.concatenate(all_band_vals)
        ymin, ymax = combined.min(), combined.max()
        pad = 0.08 * (ymax - ymin) if ymax > ymin else 0.1
        ylim: tuple[float, float] = (ymin - pad, ymax + pad)
    else:
        ylim = (-1.0, 1.0)

    sm = stability_metrics or {}

    for i, (title, color, fp_means, fp_q25, fp_q75) in enumerate(panels):
        ax = axes[0, i]
        valid = ~np.isnan(fp_means)
        x, y = bin_centers[valid], fp_means[valid]
        q25, q75 = fp_q25[valid], fp_q75[valid]

        ax.plot(x, y, color=color, linewidth=2)
        ax.fill_between(x, q25, q75, alpha=0.25, color=color, linewidth=0)
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)

        panel_title = f"{species.capitalize()} — {title}"
        if title in sm:
            rho, mad = sm[title]
            rho_str = f"{rho:.3f}" if np.isfinite(rho) else "n/a"
            mad_str = f"{mad:.3f}" if np.isfinite(mad) else "n/a"
            panel_title += f"\nρ = {rho_str},  norm. MAD = {mad_str}"
        ax.set_title(panel_title, fontsize=9)

        ax.set_xlabel(feature_label)
        ax.set_ylim(ylim)
        ax.xaxis.grid(True, linestyle="--", alpha=0.4)
        if i == 0:
            ax.set_ylabel("Mean SHAP value")

        # Observation frequency histogram inset (shared x-axis, bottom 20%)
        ax2 = ax.inset_axes((0, 0, 1.0, 0.2), zorder=0, sharex=ax, frame_on=False)
        ax2.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )
        ax2.bar(
            bin_centers, obs_counts, width=bin_widths * 0.8, color="grey", alpha=0.35
        )

    fig.suptitle(f"SHAP curve stability — {feature_label}", fontsize=11)
    fig.tight_layout()

    safe_sp = species.replace(" ", "_")
    safe_ft = feature_raw.replace("/", "_").replace(" ", "_")
    fig.savefig(
        os.path.join(figures_dir, f"shap_curve_instability_{safe_sp}_{safe_ft}.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def compute_shap_curve_stability(
    results_standard: dict[str, Any],
    results_temporal: dict[str, Any] | None,
    results_spatial: dict[str, Any] | None,
    *,
    n_bins: int = 10,
    top_n: int = 10,
    figures_dir: str = "./figures",
    key_features: list[str] | None = None,
    rho_threshold: float = 0.7,
    mad_threshold: float = 0.3,
) -> pl.DataFrame:
    """Compute SHAP dependence curve stability across CV blocking strategies.

    For each (species, top-*n* feature) pair, bins feature values into *n_bins*
    quantile bins (computed once from the full dataset under the standard baseline),
    computes the per-bin mean SHAP fingerprint for each blocking strategy, then
    measures:

    - **Spearman ρ** between the standard and alternative fingerprint vectors
      (shape consistency).
    - **Normalized MAD** = MAD of the two fingerprint vectors divided by the
      global SHAP standard deviation of that feature (scale-free magnitude
      consistency).

    Generates 3-panel dependence plots for features where Spearman ρ < *rho_threshold*
    or normalized MAD > *mad_threshold*, and unconditionally for *key_features*.

    Parameters
    ----------
    results_standard
        ``{species: ExperimentResults}`` from the standard (tree_id, no temporal)
        blocking strategy.
    results_temporal
        Same dict for temporal blocking, or ``None`` if unavailable.
    results_spatial
        Same dict for spatial (plot_id) blocking, or ``None`` if unavailable.
    n_bins
        Number of quantile bins for the dependence curve fingerprint.
    top_n
        Number of top features (by mean |SHAP| under the standard baseline) to
        include in the stability analysis per species.
    figures_dir
        Directory in which to save instability plots.
    key_features
        Feature names for which 3-panel plots are always generated regardless of
        stability metrics.
    rho_threshold
        Spearman ρ below which a feature is flagged as unstable.
    mad_threshold
        Normalized MAD above which a feature is flagged as unstable.

    Returns
    -------
    Polars DataFrame with columns: species, feature, comparison, spearman_rho,
    normalized_mad.
    """
    from scipy.stats import spearmanr
    from config import FEATURES_METADATA as _feat_meta

    os.makedirs(figures_dir, exist_ok=True)
    key_features = list(key_features or [])

    # Ordered list of (comparison_label, results_dict, color, panel_title)
    alt_strategies: list[tuple[str, dict[str, Any] | None, str, str]] = [
        (
            "temporal vs standard",
            results_temporal,
            _STRATEGY_COLORS["temporal"],
            "Temporal blocking",
        ),
        (
            "spatial vs standard",
            results_spatial,
            _STRATEGY_COLORS["spatial"],
            "Spatial blocking",
        ),
    ]

    rows: list[dict[str, object]] = []

    for species, std_res in results_standard.items():
        # ── 1. Top-n features by mean |SHAP| across all folds (standard baseline) ──
        all_shap_std = np.concatenate(
            [std_res.shap_values[fold].values for fold in range(std_res.num_folds)]
        )
        mean_abs = np.abs(all_shap_std).mean(axis=0)
        top_idxs = np.argsort(mean_abs)[::-1][:top_n]
        top_features: list[str] = [std_res.features[i] for i in top_idxs]

        extra_key = [
            f for f in key_features if f in std_res.features and f not in top_features
        ]
        all_features = top_features + extra_key

        # ── 2. Compute fingerprints for every (feature, strategy) pair ──
        # fp_cache[feature][strategy_name] = {means, q25, q75, counts}
        # fp_cache[feature]["edges"]          = the shared bin edges
        # fp_cache[feature]["global_shap_std"] = std of SHAP under standard
        fp_cache: dict[str, dict[str, Any]] = {}

        for feature in all_features:
            feat_col = std_res.X[feature].to_numpy()
            edges = _quantile_edges(feat_col, n_bins)

            # Mean fingerprint from concatenated SHAP (for stability metric comparison)
            shap_std, _ = _get_shap_arrays(std_res, feature)
            # Envelope from per-fold bin means (keeps Q25/Q75 centered on the mean)
            fp_m, fp_q25, fp_q75, fp_c = _binned_fold_envelope(std_res, feature, edges)

            fp_cache[feature] = {
                "edges": edges,
                "global_shap_std": float(np.nanstd(shap_std)) or 1.0,
                "standard": dict(means=fp_m, q25=fp_q25, q75=fp_q75, counts=fp_c),
            }

            for cmp_name, alt_results, _, _ in alt_strategies:
                if alt_results is None or species not in alt_results:
                    continue
                alt_res = alt_results[species]
                am, aq25, aq75, ac = _binned_fold_envelope(alt_res, feature, edges)
                fp_cache[feature][cmp_name] = dict(
                    means=am, q25=aq25, q75=aq75, counts=ac
                )

        # ── 3. Stability metrics from the mean-across-folds fingerprint ──
        unstable: set[str] = set()

        for feature in top_features:
            fp_std_means = fp_cache[feature]["standard"]["means"]
            gstd = fp_cache[feature]["global_shap_std"]

            for cmp_name, _, _, _ in alt_strategies:
                if cmp_name not in fp_cache[feature]:
                    continue
                fp_alt_means = fp_cache[feature][cmp_name]["means"]
                valid_mask = ~np.isnan(fp_std_means) & ~np.isnan(fp_alt_means)

                rho_val = np.nan
                mad_val = np.nan
                if valid_mask.sum() >= 3:
                    try:
                        rho_val, _ = spearmanr(
                            fp_std_means[valid_mask], fp_alt_means[valid_mask]
                        )
                    except Exception:
                        rho_val = np.nan
                    mad_val = (
                        float(
                            np.mean(
                                np.abs(
                                    fp_std_means[valid_mask] - fp_alt_means[valid_mask]
                                )
                            )
                        )
                        / gstd
                    )
                    if (
                        np.isfinite(rho_val) and rho_val < rho_threshold
                    ) or mad_val > mad_threshold:
                        unstable.add(feature)

                rows.append(
                    {
                        "species": species,
                        "feature": feature,
                        "comparison": cmp_name,
                        "spearman_rho": float(rho_val),
                        "normalized_mad": float(mad_val),
                    }
                )

        # ── 4. Generate instability and key-feature plots ──
        features_to_plot = unstable | (set(key_features) & set(std_res.features))

        for feature in features_to_plot:
            if feature not in fp_cache:
                continue

            cache = fp_cache[feature]
            edges = cache["edges"]
            gstd = cache["global_shap_std"]
            bin_centers = (edges[:-1] + edges[1:]) / 2
            bin_widths = edges[1:] - edges[:-1]

            # Clean feature label and unit from config metadata
            _meta = _feat_meta.get(feature, {})
            _label: str = str(_meta.get("label", feature))
            _unit: str | None = _meta.get("unit") or None
            feature_label: str = f"{_label} [{_unit}]" if _unit else _label

            # Observation frequency counts from the unfolded full dataset
            feat_col = std_res.X[feature].to_numpy()
            valid_obs = ~np.isnan(feat_col)
            obs_ids = np.searchsorted(edges[1:-1], feat_col[valid_obs], side="right")
            obs_counts = np.bincount(obs_ids, minlength=len(bin_centers)).astype(float)

            # Per-panel stability metrics (ρ, normalized MAD) vs. the standard baseline
            panel_metrics: dict[str, tuple[float, float]] = {}
            fp_std_m = cache["standard"]["means"]
            for cmp_name, _, _, panel_title in alt_strategies:
                if cmp_name not in cache:
                    continue
                fp_alt_m = cache[cmp_name]["means"]
                valid = ~np.isnan(fp_std_m) & ~np.isnan(fp_alt_m)
                if valid.sum() >= 3:
                    try:
                        _rho, _ = spearmanr(fp_std_m[valid], fp_alt_m[valid])
                    except Exception:
                        _rho = np.nan
                    _mad = (
                        float(np.mean(np.abs(fp_std_m[valid] - fp_alt_m[valid]))) / gstd
                    )
                    panel_metrics[panel_title] = (float(_rho), float(_mad))

            # Build panels: standard first, then available comparisons in order
            panels: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]] = [
                (
                    "Standard baseline",
                    _STRATEGY_COLORS["standard"],
                    cache["standard"]["means"],
                    cache["standard"]["q25"],
                    cache["standard"]["q75"],
                ),
            ]
            for cmp_name, _, color, panel_title in alt_strategies:
                if cmp_name in cache:
                    d = cache[cmp_name]
                    panels.append((panel_title, color, d["means"], d["q25"], d["q75"]))

            _plot_shap_stability_figure(
                species,
                feature,
                feature_label,
                panels,
                bin_centers,
                bin_widths,
                obs_counts,
                figures_dir,
                panel_metrics,
            )

    if not rows:
        return pl.DataFrame(
            schema={
                "species": pl.Utf8,
                "feature": pl.Utf8,
                "comparison": pl.Utf8,
                "spearman_rho": pl.Float64,
                "normalized_mad": pl.Float64,
            }
        )

    return pl.DataFrame(rows).select(
        pl.col("species").cast(pl.Utf8),
        pl.col("feature").cast(pl.Utf8),
        pl.col("comparison").cast(pl.Utf8),
        pl.col("spearman_rho").cast(pl.Float64),
        pl.col("normalized_mad").cast(pl.Float64),
    )


def plot_residuals_histogram(
    results: ExperimentResults,
    fold: int | None = None,
    split: Split = "test",
    original_space: bool = False,
    bins: int | str = "auto",
    ax: Axes | None = None,
    color: str = "#1f77b4",
) -> Axes:
    """Plot a histogram of out-of-sample prediction residuals (y_true − y_pred).

    Parameters
    ----------
    results
        Results object containing predictions and true values.
    fold
        Fold index to use. If None, residuals from all folds are concatenated.
    split
        Data split to use ('train', 'test', or 'all'). Defaults to 'test'.
    original_space
        If True and dist_params is available, residuals are computed in the
        original growth-rate space (%). Otherwise quantile space is used.
    bins
        Number of bins or binning strategy passed to ax.hist.
    ax
        Axes to plot on. If None, a new figure is created.
    color
        Histogram and KDE line color.

    Returns
    -------
    The axes object with the residual histogram.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    folds = [fold] if fold is not None else list(range(results.num_folds))

    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    for f in folds:
        _, y_true_s, y_pred_s = results.get_data(f, split)
        if original_space and results.dist_params is not None:
            y_true_np = results.get_inverse_transform(
                y_true_s, results.dist_params
            ).to_numpy()
            y_pred_np = results.get_inverse_transform(
                y_pred_s, results.dist_params
            ).to_numpy()
        else:
            y_true_np = y_true_s.to_numpy()
            y_pred_np = y_pred_s.to_numpy()
        y_true_parts.append(y_true_np)
        y_pred_parts.append(y_pred_np)

    y_true_all = np.concatenate(y_true_parts)
    y_pred_all = np.concatenate(y_pred_parts)
    residuals = y_true_all - y_pred_all

    in_orig = original_space and results.dist_params is not None
    if not in_orig:
        residuals = residuals * 100.0

    rmse = float(np.sqrt(np.mean(residuals**2)))
    mean = float(np.mean(residuals))
    skew = float(_scipy_stats.skew(residuals))

    ax.hist(residuals, bins=bins, color=color, alpha=0.5, density=True)
    sns.kdeplot(residuals, ax=ax, color=color, linewidth=1.5)
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)

    stats_text = f"RMSE  = {rmse:.2f}%\nMean  = {mean:.2f}%\nSkew  = {skew:.2f}"
    ax.text(
        0.97,
        0.97,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        family="monospace",
    )

    ax.set_title(results.species.capitalize())

    ax.set_xlabel("Residual [%/yr]" if in_orig else "Residual [pct. rank %]")
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    # Use scientific notation for y-axis
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Density")

    return ax
