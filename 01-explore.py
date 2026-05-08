import marimo

__generated_with = "0.23.2"
app = marimo.App()


@app.cell
def _():
    import polars as pl
    import polars.selectors as cs
    from typing import get_args

    from data import load_data
    from config import Species, TARGET, FEATURES_METADATA

    # Load data for the given species
    df = pl.concat(
        [load_data(species) for species in get_args(Species)], how="vertical_relaxed"
    )

    df.describe()
    return FEATURES_METADATA, TARGET, cs, df, pl


@app.cell
def _(cs, df, pl):
    # Total number of rows and rows with defoliation / soil solution data
    print("Number of rows in total:", df.height)
    height = df.filter(pl.any_horizontal(cs.starts_with("dep_").is_not_null())).height
    print("Number of rows with defoliation data:", height)
    height = df.filter(pl.any_horizontal(cs.starts_with("ss_").is_not_null())).height
    print("Number of rows with soil solution data:", height)
    species = df.select(pl.col("species").unique()).to_series()
    # Total number of trees, plots in total and species
    print("Number of unique trees:", df.select(pl.col("tree_id").n_unique()).item())
    print("Number of unique plots:", df.select(pl.col("plot_id").n_unique()).item())
    for _sp in species:
        n_trees = (
            df.filter(pl.col("species") == _sp)
            .select(pl.col("tree_id").n_unique())
            .item()
        )
        n_plots = (
            df.filter(pl.col("species") == _sp)
            .select(pl.col("plot_id").n_unique())
            .item()
        )
        print(f"- {_sp.capitalize()}: {n_trees} trees, {n_plots} plots")
    return


@app.cell
def _(df, pl):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats

    species_list = df["species"].unique().to_list()
    n = len(species_list)
    ncols = 2
    nrows = (n + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10))
    axes = axes.flatten()
    for i, _sp in enumerate(["spruce", "pine", "beech", "oak"]):
        ax = axes[i]
        values = (
            df.filter(pl.col("species") == _sp)
            .select(pl.col("growth_rate_rel") * 100)
            .to_series()
            .drop_nulls()
            .to_numpy()
        )
        values = values[values > 0]
        shape, loc, scale = stats.lognorm.fit(values)
        mu = np.log(scale)
        sigma = shape
        ks_stat, _ = stats.kstest(values, "lognorm", args=(shape, loc, scale))
        x_min, x_max = (0, 6)
        sns.histplot(
            values,
            bins=np.arange(x_min, x_max, 0.1),
            ax=ax,
            stat="density",
            color="steelblue",
            alpha=0.4,
            edgecolor="white",
            linewidth=0.5,
        )
        x = np.linspace(x_min, x_max, 500)
        pdf = stats.lognorm.pdf(x, shape, loc, scale)
        ax.plot(x, pdf, color="crimson", linewidth=2, label="Log-normal fit")
        ax.set_title(f"{_sp.capitalize()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Relative growth rate [%/year]")
        ax.set_ylabel("Density")  # Fit log-normal
        ax.set_xlim(x_min, x_max)
        ax.legend(fontsize=8)
        ax_inset = ax.inset_axes([0.35, 0.3, 0.6, 0.6])
        log_values = np.log(values - loc)
        (osm, osr), (slope, intercept, r) = stats.probplot(
            log_values, dist="norm"
        )  # KS statistic only
        ax_inset.scatter(osm, osr, s=2, alpha=0.4, color="steelblue", rasterized=True)
        ax_inset.plot(
            osm, slope * np.array(osm) + intercept, color="crimson", linewidth=1
        )
        ax_inset.set_title("Q-Q (log)", fontsize=7)
        ax_inset.set_xlabel("Theoretical [z-score]", fontsize=6)
        ax_inset.set_ylabel(
            "Sample [log(%/year)]", fontsize=6
        )  # --- Main histogram ---
        ax_inset.tick_params(labelsize=5)
        textstr = f"$\\mu={mu:.2f}$, $\\theta={loc:.2f}$, $\\sigma={sigma:.2f}$\n$R^2={r**2:.3f}$\nKS stat = {ks_stat:.3f}"
        print(
            f"{_sp}: mu={mu:.2f}, sigma={sigma:.2f}, loc={loc:.2f}, KS stat={ks_stat:.3f}, R^2={r**2:.3f}"
        )
        ax_inset.text(
            0.935,
            0.45,
            textstr,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    # Hide unused subplots
    plt.show()  # --- Inset Q-Q plot (top-right, inside histogram) ---  # [x0, y0, width, height] in axes coords
    return plt, sns


@app.cell
def _(FEATURES_METADATA, pl):
    pl.from_dicts(
        [
            {**{"feature": feature}, **descr}
            for feature, descr in FEATURES_METADATA.items()
        ]
    )
    return


@app.cell
def _(df):
    # Compare absolute growth, growth rate, and relative growth rate (to diameter)
    df.select("growth", "growth_rate", "growth_rate_rel").describe()
    return


@app.cell
def _(FEATURES_METADATA, df):
    # Check that data contains all features
    missing_features = set(FEATURES_METADATA.keys()) - set(df.columns)
    if missing_features:
        raise ValueError(
            f"Data is missing the following features: {missing_features}. "
            "Please check the data loading process."
        )
    return


@app.cell
def _(df, pl, plt, sns):
    print("Distribution of the number of trees per plot:")
    num_trees = df.group_by("plot_id").agg(pl.count("tree_id").alias("num_trees"))
    print(f"# min = {num_trees['num_trees'].min()}")
    # Distribution of the number of trees per plot
    print(f"# max = {num_trees['num_trees'].max()}")
    print(f"# mean = {num_trees['num_trees'].mean()}")
    print(
        f"# of single-tree plots = {len(num_trees.filter(num_trees['num_trees'] == 1))}"
    )
    _ = sns.histplot(num_trees["num_trees"], bins=20)
    plt.xlabel("# of trees")
    plt.ylabel("# of plots")
    plt.title("Distribution of the number of trees per plot")
    return


@app.cell
def _(TARGET, df, pl, plt, sns):
    # Plot box plots of target variable by plot_id

    # Keep only plots with at least 10 trees
    data = df.with_columns(
        pl.col("tree_id").n_unique().over("plot_id").alias("num_trees")
    ).filter(pl.col("num_trees") >= 100)
    sns.boxplot(x="plot_id", y=TARGET, data=data.to_pandas())
    plt.xlabel("Plot ID")

    # Vertical label for x-axis
    _ = plt.xticks(rotation=90)
    return


@app.cell
def _(df, plt, sns):
    # Plot distribution of trees of latitude and longitude
    plt.figure(figsize=(5, 4))
    sns.histplot(df["plot_latitude"], bins=20)
    plt.xlabel("Latitude")
    plt.ylabel("# of trees")
    plt.title("Distribution of latitude")

    # Plot distribution of trees of altitudes
    plt.figure(figsize=(5, 4))
    sns.histplot(df["plot_altitude"], bins=20)
    plt.xlabel("Altitude")
    plt.ylabel("# of trees")
    plt.title("Distribution of altitude")

    # Plot distributions of trees across plot orientation for each species
    plt.figure(figsize=(5, 4))
    sns.histplot(
        data=df.to_pandas(),
        x="plot_orientation",
        bins=20,
        hue="species",
        multiple="stack",
        stat="count",
    )
    plt.xlabel("Orientation")
    plt.ylabel("# of trees")
    plt.title("Distribution of plot orientation")
    plt.xticks(rotation=90)
    plt.show()
    return


@app.cell
def _(df, pl, plt, sns):
    # plot distribution of plots across orientations
    plt.figure(figsize=(5, 4))
    sns.histplot(
        data=df.group_by("species", "plot_id").agg(
            pl.first("plot_orientation").alias("plot_orientation")
        ),
        x="plot_orientation",
        bins=20,
        hue="species",
        multiple="dodge",
        shrink=0.8,
    )
    plt.xlabel("Orientation")
    plt.ylabel("# of plots")
    plt.title("Distribution of plots across orientations")
    plt.xticks(rotation=90)
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
