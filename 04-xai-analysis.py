import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # XAI Analysis

    Loads pre-computed results from `./cache/` (run `./train-all.sh` first).
    Select experiment configuration below — the notebook updates reactively.
    """)
    return


@app.cell
def _():
    import joblib
    import numpy as np
    import os

    import matplotlib.pyplot as plt
    import seaborn as sns
    import polars as pl
    import polars.selectors as cs

    from models import ALL_SPECIES
    from config import FEATURES_METADATA
    from explain import plot_dependence, compute_interaction_matrix

    return (
        ALL_SPECIES,
        FEATURES_METADATA,
        compute_interaction_matrix,
        cs,
        joblib,
        np,
        os,
        pl,
        plot_dependence,
        plt,
        sns,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Global Performance
    """)
    return


@app.cell
def _(joblib, mo, os, pl):
    import re as _re
    import glob as _glob
    from analysis import summarize_performance as _summarize

    # Filename pattern: results-{ablation}-{model_type}-{group_col}.pkl
    # ablation may contain hyphens, so anchor model_type and group_col from the right.
    _pat = _re.compile(r"^results-(.+)-(gbdt|elasticnet)-(tree_id|plot_id|None)\.pkl$")
    _csv = "./cache/performance_summary.csv"

    _loaded = []
    for _path in sorted(_glob.glob("./cache/results-*.pkl")):
        _m = _pat.match(os.path.basename(_path))
        if not _m:
            continue
        _ablation, _model_type, _group_col_str = _m.groups()
        _results = joblib.load(_path)
        _summarize(
            _results,
            ablation=_ablation,
            model_type=_model_type,
            group_col=_group_col_str,
            use_temporal_cv=True,
        )
        _loaded.append(f"{_model_type}/{_ablation}/{_group_col_str}")

    mo.stop(
        not _loaded,
        mo.callout(
            mo.md("No cached results found in `./cache/`. Run `./train-all.sh` first."),
            kind="warn",
        ),
    )

    perf_df = pl.read_csv(_csv).filter(pl.col("temporal_cv") == "yes")

    mo.md(
        f"Loaded **{len(_loaded)}** cached run(s): {', '.join(f'`{r}`' for r in _loaded)}"
    )
    return (perf_df,)


@app.cell
def _(mo, perf_df, pl):
    _ABLATION_LABELS = {
        "all": "all features",
        "no-defoliation": "no defoliation",
        "plot-level-only": "plot-level features only",
        "tree-level-only": "tree-level features only",
    }
    _ABLATION_ORDER = {
        "all": 0,
        "no-defoliation": 1,
        "plot-level-only": 2,
        "tree-level-only": 3,
    }
    _MODEL_LABELS = {"gbdt": "GBDT", "elasticnet": "ElasticNet"}

    def build_paper_table(df: pl.DataFrame) -> pl.DataFrame:
        """Format a filtered performance DataFrame into a paper-ready R² table."""
        _r2_rows = df.filter(pl.col("split") == "test_r2").select(
            "model", "ablation", "spruce", "pine", "beech", "oak"
        )
        # Weighted mean R² = Σ_species (n_species/n_total × R²_species)
        _weighted = df.filter(pl.col("split") == "test_weight_r2").select(
            "model",
            "ablation",
            weighted_r2=(
                pl.col("spruce").cast(pl.Float64, strict=False).fill_null(0.0)
                + pl.col("pine").cast(pl.Float64, strict=False).fill_null(0.0)
                + pl.col("beech").cast(pl.Float64, strict=False).fill_null(0.0)
                + pl.col("oak").cast(pl.Float64, strict=False).fill_null(0.0)
            )
            .round(2)
            .cast(pl.Utf8),
        )
        return (
            _r2_rows.join(_weighted, on=["model", "ablation"], how="left")
            .with_columns(
                config=pl.concat_str(
                    pl.col("model").map_elements(
                        lambda m: _MODEL_LABELS.get(m, m), return_dtype=pl.Utf8
                    ),
                    pl.lit(" ("),
                    pl.col("ablation").map_elements(
                        lambda a: _ABLATION_LABELS.get(a, a), return_dtype=pl.Utf8
                    ),
                    pl.lit(")"),
                ),
                _model_ord=pl.when(pl.col("model") == "gbdt").then(0).otherwise(1),
                _ablation_ord=pl.col("ablation").map_elements(
                    lambda a: _ABLATION_ORDER.get(a, 99), return_dtype=pl.Int32
                ),
            )
            .sort("_model_ord", "_ablation_ord")
            .rename(
                {
                    "spruce": "Spruce",
                    "pine": "Pine",
                    "beech": "Beech",
                    "oak": "Oak",
                    "weighted_r2": "Weighted R²",
                }
            )
            .select("config", "Spruce", "Pine", "Beech", "Oak", "Weighted R²")
            .rename({"config": "Configuration"})
        )

    _df_tree = perf_df.filter(pl.col("group_by") == "tree_id")
    mo.vstack(
        [
            mo.md(
                "**Table 2**: R² test scores on 5-fold cross-validation grouped by tree identifiers."
            ),
            mo.ui.table(build_paper_table(_df_tree)),
        ]
    )
    return (build_paper_table,)


@app.cell
def _(build_paper_table, mo, perf_df, pl):
    _df_plot = perf_df.filter(
        (pl.col("group_by") == "plot_id") & (pl.col("ablation") == "all")
    )

    mo.stop(
        _df_plot.is_empty(),
        mo.callout(
            mo.md(
                "No `plot_id` results found. "
                "Run `uv run train.py --group-col plot_id --temporal-cv` for each model type."
            ),
            kind="info",
        ),
    )

    mo.vstack(
        [
            mo.md(
                "**Table 3**: R² test scores on 5-fold cross-validation grouped by plot identifiers. "
                "Best strictly positive score for each species is indicated in bold."
            ),
            mo.ui.table(build_paper_table(_df_plot)),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Feature importance
    """)
    return


@app.cell
def _(mo):
    model_type_ui = mo.ui.dropdown(["gbdt", "elasticnet"], value="gbdt", label="Model")
    ablation_ui = mo.ui.dropdown(
        ["all", "no-defoliation", "tree-level-only", "plot-level-only"],
        value="all",
        label="Ablation",
    )
    group_col_ui = mo.ui.dropdown(
        ["tree_id", "plot_id", "none"], value="tree_id", label="Group by"
    )
    temporal_cv_ui = mo.ui.switch(value=True, label="Temporal CV")
    weight_shap_ui = mo.ui.switch(value=True, label="Weight SHAP by n")

    mo.hstack(
        [model_type_ui, ablation_ui, group_col_ui, temporal_cv_ui, weight_shap_ui],
        gap=2,
    )
    return ablation_ui, group_col_ui, model_type_ui, weight_shap_ui


@app.cell
def _(
    ablation_ui,
    group_col_ui,
    joblib,
    mo,
    model_type_ui,
    os,
    weight_shap_ui,
):
    model_type = model_type_ui.value
    ablation = ablation_ui.value
    group_col = None if group_col_ui.value == "none" else group_col_ui.value
    weight_shap_fimp = weight_shap_ui.value

    _results_path = f"./cache/results-{ablation}-{model_type}-{group_col}.pkl"
    mo.stop(
        not os.path.exists(_results_path),
        mo.callout(
            mo.md(
                f"No cached results at `{_results_path}`.\n\nRun `./train-all.sh` (or "
                f"`uv run train.py --model-type {model_type} --ablation {ablation} "
                f"--group-col {group_col or 'none'} --temporal-cv`) first."
            ),
            kind="warn",
        ),
    )
    all_results = joblib.load(_results_path)
    mo.md(f"Loaded **{len(all_results)} species** from `{_results_path}`")
    return ablation, all_results, group_col, model_type, weight_shap_fimp


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Hyperparameters
    """)
    return


@app.cell
def _(ablation, group_col, mo, model_type, os, pl):
    _hp_path = f"./cache/hyperparams-{ablation}-{model_type}-{group_col}.parquet"

    mo.stop(
        not os.path.exists(_hp_path),
        mo.callout(
            mo.md(
                f"`{_hp_path}` not found — re-run training to generate hyperparameter logs."
            ),
            kind="info",
        ),
    )

    _hp = pl.read_parquet(_hp_path)
    _param_cols = [c for c in _hp.columns if c not in ("species", "fold")]

    hp_summary = (
        _hp.group_by("species")
        .agg(
            [
                pl.concat_str(
                    pl.col(p).mean().round(4).cast(pl.Utf8),
                    pl.lit(" ± "),
                    pl.col(p).std().round(4).cast(pl.Utf8),
                ).alias(p)
                for p in _param_cols
            ]
        )
        .sort("species")
    )

    mo.vstack(
        [
            mo.md("Mean ± std across CV folds per species."),
            mo.ui.table(hp_summary),
            mo.md("**Per-fold values**"),
            mo.ui.table(_hp.sort("species", "fold")),
        ]
    )
    return


@app.cell
def _(ablation, all_results, cs, group_col, model_type, np, pl):
    feature_importances = pl.from_dicts(
        [
            {
                "species": _sp,
                "fold": _fold,
                **dict(
                    zip(
                        _res.features,
                        np.abs(_res.shap_values[_fold].values).mean(axis=0),
                    )
                ),
                "n": len(_res.shap_values[_fold].values),
            }
            for _sp, _res in all_results.items()
            for _fold in range(_res.num_folds)
        ]
    ).unpivot(
        on=cs.exclude("species", "fold", "n"),
        index=["species", "fold", "n"],
        variable_name="feature",
        value_name="shap",
    )

    feature_importances.write_parquet(
        f"./cache/feature_importances-{ablation}-{model_type}-{group_col}.parquet"
    )
    feature_importances
    return (feature_importances,)


@app.cell
def _(FEATURES_METADATA, feature_importances, pl, weight_shap_fimp):
    _min_importance = 1.0
    _max_rank = 8

    _base = (
        feature_importances.with_columns(
            importance=pl.col("shap").mean().over("species", "feature")
        )
        .with_columns(
            rank=pl.col("importance")
            .rank(descending=True, method="dense")
            .over("species")
        )
        .filter(
            (pl.col("importance").max().over("feature") >= _min_importance)
            | (pl.col("rank").min().over("feature") <= _max_rank)
        )
        .join(
            pl.from_dicts(
                [
                    {"feature": k, "feature_label": v["label"]}
                    for k, v in FEATURES_METADATA.items()
                ]
            ),
            on="feature",
            how="left",
        )
        .with_columns(feature=pl.col("feature_label"))
        .drop("feature_label")
        .sort(
            "species",
            pl.col("importance").mean().over("feature"),
            descending=[False, True],
        )
    )

    if weight_shap_fimp:
        _agg = (
            _base.group_by("feature", "fold")
            .agg(
                shap=(pl.col("shap") * pl.col("n")).sum() / pl.col("n").sum(),
                importance=(pl.col("importance") * pl.col("n")).sum()
                / pl.col("n").sum(),
                n=pl.col("n").sum(),
            )
            .with_columns(species=pl.lit("all species"))
            .with_columns(
                rank=pl.col("importance")
                .rank(descending=True, method="dense")
                .over("species")
            )
            .select(_base.columns)
        )
    else:
        _agg = (
            _base.group_by("feature", "fold")
            .agg(
                shap=pl.col("shap").mean(),
                importance=pl.col("importance").mean(),
                n=pl.col("n").sum(),
            )
            .with_columns(species=pl.lit("all species"))
            .with_columns(
                rank=pl.col("importance")
                .rank(descending=True, method="dense")
                .over("species")
            )
            .select(_base.columns)
        )

    fimp_data = pl.concat([_base, _agg], how="vertical_relaxed")
    fimp_data
    return (fimp_data,)


@app.cell
def _(
    ablation,
    feature_importances,
    fimp_data,
    group_col,
    model_type,
    pl,
    plt,
    sns,
    weight_shap_fimp,
):
    _n_species = feature_importances.select("species").n_unique()

    g = sns.catplot(
        fimp_data.with_columns((pl.col("shap") * 100).alias("shap_percent")),
        x="shap_percent",
        y="feature",
        hue="species",
        kind="bar",
        palette=sns.color_palette("cmo.thermal", n_colors=_n_species + 1),
        height=8,
        aspect=0.6,
    )

    g._legend.set_title("Species")
    for _lbl in g._legend.texts:
        _lbl.set_text(_lbl.get_text().capitalize())

    _ax = g.facet_axis(0, 0)
    _new_ylabels = []
    for _ytick, _ylabel in zip(_ax.get_yticks(), _ax.get_yticklabels()):
        _fname = _ylabel.get_text()
        _row = (
            fimp_data.group_by("feature", "species")
            .agg(pl.col("rank").mean())
            .filter(
                (pl.col("feature") == _fname) & (pl.col("species") == "all species")
            )
            .select("rank")
        )
        _new_ylabels.append(
            f"({int(_row.item())}) {_fname}" if len(_row) > 0 else _fname
        )
    _ax.set_yticklabels(_new_ylabels)

    plt.xlabel(
        "Weighted feature importance (mean |SHAP| %)"
        if weight_shap_fimp
        else "Feature importance (mean |SHAP| %)"
    )
    plt.ylabel("Feature")
    plt.title(
        f"Feature importance ({'GBDT' if model_type == 'gbdt' else 'ElasticNet'}, "
        f"{'all features' if ablation == 'all' else 'w/o defoliation'})"
    )
    plt.savefig(
        f"./figures/importance-{model_type}-{group_col}-{ablation}.pdf",
        bbox_inches="tight",
    )
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Ablation comparison (all vs. no-defoliation)
    """)
    return


@app.cell
def _(ablation, feature_importances, group_col, mo, model_type, os, pl):
    _other = "no-defoliation" if ablation == "all" else "all"
    _other_path = (
        f"./cache/feature_importances-{_other}-{model_type}-{group_col}.parquet"
    )

    mo.stop(
        not os.path.exists(_other_path),
        mo.callout(
            mo.md(f"Comparison skipped — `{_other_path}` not found."),
            kind="info",
        ),
    )

    importances_comparison = (
        feature_importances.group_by("species", "feature")
        .agg(pl.col("shap").mean().alias(f"shap-{ablation}") * 100)
        .with_columns(
            pl.col(f"shap-{ablation}")
            .rank(descending=True)
            .over("species")
            .cast(pl.Int32)
            .alias(f"rank-{ablation}")
        )
        .join(
            pl.read_parquet(_other_path)
            .group_by("species", "feature")
            .agg(pl.col("shap").mean().alias(f"shap-{_other}") * 100)
            .with_columns(
                pl.col(f"shap-{_other}")
                .rank(descending=True)
                .over("species")
                .cast(pl.Int32)
                .alias(f"rank-{_other}")
            ),
            on=["species", "feature"],
            how="full",
            validate="1:1",
            coalesce=True,
        )
    )
    importances_comparison
    return (importances_comparison,)


@app.cell
def _(ablation, importances_comparison, pl):
    _other = "no-defoliation" if ablation == "all" else "all"
    _top_n = 3

    top_changes = (
        importances_comparison.with_columns(
            shap_delta=pl.col(f"shap-{ablation}") - pl.col(f"shap-{_other}"),
            rank_delta=pl.col(f"rank-{_other}") - pl.col(f"rank-{ablation}"),
        )
        .filter(
            (
                pl.col("shap_delta").rank("dense", descending=True).over("species")
                <= _top_n
            )
            & (pl.col(f"rank-{ablation}") <= 10)
        )
        .sort(["species", "shap_delta"], descending=[False, True])
    )

    with pl.Config() as _cfg:
        _cfg.set_tbl_formatting("ASCII_MARKDOWN")
        _cfg.set_float_precision(1)
        _cfg.set_tbl_rows(100)
        _cfg.set_tbl_hide_column_data_types(True)
        print(
            top_changes.sort(
                by=["species", f"rank-{_other}"], descending=[False, False]
            )
        )

    top_changes
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Dependence plots
    """)
    return


@app.cell(hide_code=True)
def _(ALL_SPECIES, FEATURES_METADATA, mo):
    dep_feature_ui = mo.ui.dropdown(
        list(FEATURES_METADATA.keys()),
        value="defoliation_mean",
        label="Feature",
    )
    dep_species_ui = mo.ui.dropdown(
        ALL_SPECIES + ["all"],
        value="all",
        label="Species",
    )
    mo.hstack([dep_feature_ui, dep_species_ui], gap=2)
    return dep_feature_ui, dep_species_ui


@app.cell
def _(
    ALL_SPECIES,
    all_results,
    dep_feature_ui,
    dep_species_ui,
    plot_dependence,
    plt,
):
    _feature = dep_feature_ui.value
    _species_sel = dep_species_ui.value
    _species_list = ALL_SPECIES if _species_sel == "all" else [_species_sel]

    _n = len(_species_list)
    _fig, _axes = plt.subplots(
        (_n + 1) // 2, min(_n, 2), figsize=(12, 4 * ((_n + 1) // 2)), squeeze=False
    )

    for _i, (_sp, _ax) in enumerate(zip(_species_list, _axes.flatten())):
        plot_dependence(
            all_results[_sp],
            feature=_feature,
            alpha=0.3,
            ax=_ax,
        )
        _ax.set_title(_sp.capitalize())
        _ax.set_xlabel(_feature)
        _ax.set_ylabel("SHAP value [%]")

    for _j in range(_i + 1, len(_axes.flatten())):
        _axes.flatten()[_j].set_visible(False)

    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Feature interactions
    """)
    return


@app.cell(hide_code=True)
def _(ALL_SPECIES, mo):
    interactions_species_ui = mo.ui.dropdown(
        ALL_SPECIES, value="spruce", label="Species"
    )
    interactions_species_ui
    return (interactions_species_ui,)


@app.cell
def _(
    all_results,
    compute_interaction_matrix,
    feature_importances,
    interactions_species_ui,
    pl,
    plt,
):
    _sp = interactions_species_ui.value
    _top_n_features = (
        feature_importances.select(
            "feature", pl.col("shap").mean().over("feature").alias("importance")
        )
        .unique()
        .sort("importance", descending=True)
        .head(20)["feature"]
        .to_list()
    )

    plt.figure(figsize=(10, 8))
    _ax = plt.gca()
    compute_interaction_matrix(
        all_results[_sp], top_n=_top_n_features, ax=_ax, vmax=0.006
    )
    plt.title(f"Feature interactions — {_sp.capitalize()}")
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
