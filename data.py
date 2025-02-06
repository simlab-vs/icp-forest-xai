from config import DATA_PATH, Species, TARGET, FEATURES, CATEGORICAL_COLUMNS

import numpy as np
import polars as pl
import os
import matplotlib.pyplot as plt


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
    if species == "spruce":  # Norway Spruce
        query = "specie == 'Picea abies'"
    elif species == "pine":  # Scots Pine
        query = "specie == 'Pinus sylvestris'"
    elif species == "beech":  # Common Beech
        query = "specie == 'Fagus sylvatica'"
    elif species == "oak":  # Sessile & Pedunculate Oak
        query = "(specie == 'Quercus petraea') | (specie == 'Quercus robur')"

    query = f"SELECT * FROM self WHERE {query}"

    return pl.read_parquet(
        os.path.join(DATA_PATH, "tidy", "cpf-level2_growth-periods_with-cc.parquet")
    ).sql(query)


def prepare_data(
    df: pl.DataFrame, restrict_features: list[str] | None = None, plotting: bool = False
) -> tuple[pl.DataFrame, pl.Series]:
    """Prepare the data for training.

    We normalize the target variable by fitting a log-normal distribution to it and
    transforming it to quantiles of the fitted distribution.

    Parameters
    ----------
    df
        Dataframe containing the data.
    restrict_features
        Features to include in the model, if None all features are included.
    plotting
        Whether to plot the fitted distribution and Q-Q plot.

    Returns
    -------
    A tuple containing the features and the transformed target.
    """
    # Fit a log-normal distribution to the target
    from scipy.stats import lognorm, kstest, probplot

    y = df[TARGET]

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
    y_quantiles = lognorm.cdf(y_plus_one, shape, loc, scale)
    y_quantiles = pl.Series(y_quantiles, dtype=pl.Float64)

    # Select the features and the transformed target
    if restrict_features:
        df = df.select(restrict_features)
    else:
        df = df.select(FEATURES)

    return cat_to_codes(df, CATEGORICAL_COLUMNS), y_quantiles


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
