from config import (
    DATA_PATH,
    Species,
    TARGET,
    CATEGORICAL_COLUMNS,
    Ablation,
    FEATURES_DESCRIPTION,
)

import numpy as np
import polars as pl
import os
import re

import matplotlib.pyplot as plt

from typing import Sequence

# spell-checker: disable
SPECIES_MAPPING = {
    "Picea abies": "spruce",  # Norway Spruce
    "Pinus sylvestris": "pine",  # Scots Pine
    "Fagus sylvatica": "beech",  # Common Beech
    "Quercus petraea": "oak",  # Sessile Oak
    "Quercus robur": "oak",  # Pedunculate Oak
}


def perform_ablation(ablation: Ablation, features: Sequence[str]) -> Sequence[str]:
    """
    Perform ablation on the features based on the specified ablation type.

    Parameters
    ----------
    - ablation (Ablation): The type of ablation to perform.
    - features (list[str]): The list of features to ablate.

    Returns
    -------
    The ablated list of features.
    """

    if ablation == "all":
        return features
    elif ablation == "tree-level-only":
        return [
            feature
            for feature in features
            if FEATURES_DESCRIPTION[feature]["level"] == "tree"
        ]
    elif ablation == "plot-level-only":
        return [
            feature
            for feature in features
            if FEATURES_DESCRIPTION[feature]["level"] == "plot"
        ]
    elif (match := re.match(r"(.*)-defoliation", ablation)) is not None:
        prefix = match.group(1)

        if prefix == "no":
            return [feature for feature in features if "defoliation" not in feature]
        else:
            return [
                feature
                for feature in features
                if "defoliation" not in feature or feature == f"defoliation_{prefix}"
            ]
    else:
        raise ValueError(
            f"Unknown ablation type: {ablation}. "
            f"Expected one of: 'all', 'tree-level-only', 'plot-level-only', 'no-defoliation'."
        )


def load_data(species: Species) -> pl.DataFrame:
    """Load data for the given species.

    Parameters
    ----------
    species
        Species to load data for.

    Returns
    -------
    Data for the given species.
    """
    return (
        pl.read_parquet(
            # os.path.join(DATA_PATH, "tidy", "cpf-level2_growth-periods_with-cc.parquet")
            os.path.join(DATA_PATH, "tidy", "cpf-level2_cleaned.parquet")
        )
        .with_columns(species=pl.col("specie").cast(pl.Utf8).replace(SPECIES_MAPPING))
        .filter(pl.col("species") == species)
    )


def prepare_data(
    df: pl.DataFrame, ablation: Ablation = "all", plotting: bool = False
) -> tuple[pl.DataFrame, pl.Series]:
    """Prepare the data for training.

    We normalize the target variable by fitting a log-normal distribution to it and
    transforming it to quantiles of the fitted distribution.

    Parameters
    ----------
    df
        Dataframe containing the data.
    ablation
        Ablation to apply to the data. If "all", no ablation is applied.
    plotting
        Whether to plot the fitted distribution and Q-Q plot.

    Returns
    -------
    A tuple containing the features and the transformed target.
    """
    # Fit a log-normal distribution to the target
    from scipy.stats import lognorm, kstest, probplot

    # If we have predicted defoliation, calculate the residuals
    if "defoliation_mean_pred" in df.columns:
        df = df.with_columns(
            defoliation_mean_residual=df["defoliation_mean"]
            - df["defoliation_mean_pred"]
        )

    # Select the features and the transformed target
    X = df.select(perform_ablation(ablation, list(FEATURES_DESCRIPTION.keys())))

    X = cat_to_codes(X, CATEGORICAL_COLUMNS).fill_nan(None)
    y = df[TARGET]

    # Apply a log-normal transformation to the target if it is growth rate
    if TARGET != "growth_rate_rel":
        return X, y

    y_plus_one = y + 1.0
    shape, loc, scale = lognorm.fit(y_plus_one)

    # Perform the Kolmogorov-Smirnov test
    ks_stat, p_value = kstest(y_plus_one, "lognorm", args=(shape, loc, scale))

    print(f"KS Statistic: {ks_stat}")
    print(f"P-value: {p_value:0.2}")

    if plotting:
        # Plot the data and the fitted distribution for visualization
        x = np.linspace(min(y_plus_one), max(y_plus_one), 1000)
        pdf_fitted = lognorm.pdf(x, shape, loc, scale)
        _, bins = np.histogram(y_plus_one, bins=30, density=True)

        plt.figure(figsize=(10, 6))
        plt.hist(
            y_plus_one,
            bins=bins.tolist(),
            density=True,
            alpha=0.6,
            color="skyblue",
            label="Data Histogram",
        )
        plt.plot(x, pdf_fitted, "r-", label="Fitted Log-Normal PDF")
        plt.title("Goodness-of-Fit: Log-Normal Distribution")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()

        # Plot the Q-Q plot
        plt.figure(figsize=(10, 6))
        probplot(y_plus_one, dist="lognorm", sparams=(shape, loc, scale), plot=plt)
        plt.title("Q-Q Plot: Log-Normal Distribution")

    # Transform the target to quantiles of the fitted distribution
    y_log_norm = lognorm.cdf(y_plus_one, shape, loc, scale)
    y_log_norm = pl.Series(y_log_norm, dtype=pl.Float64)

    return X, y_log_norm


def cat_to_codes(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Convert categorical columns to codes"""
    for col in cols:
        # Skip columns that are not present in the dataframe
        if col not in df.columns:
            continue

        df = df.with_columns(
            pl.col(col).cast(pl.Categorical).to_physical().cast(pl.Int64)
        )

    return df  #
