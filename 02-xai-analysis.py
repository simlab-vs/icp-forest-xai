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
    from explain import plot_dependence, compute_interaction_matrix, PlotType

    return (
        ALL_SPECIES,
        FEATURES_METADATA,
        PlotType,
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
    _pat = _re.compile(
        r"^results-(.+)-(gbdt|elasticnet|lmm)-(tree_id|plot_id|None)-(temporal|standard)\.pkl$"
    )
    _csv = "./cache/performance_summary.csv"

    # Remove if
    if os.path.exists(_csv):
        os.remove(_csv)

    _loaded = []
    for _path in sorted(_glob.glob("./cache/results-*.pkl")):
        _m = _pat.match(os.path.basename(_path))
        if not _m:
            continue
        _ablation, _model_type, _group_col_str, _tcv_str = _m.groups()
        _results = joblib.load(_path)
        _summarize(
            _results,
            ablation=_ablation,
            model_type=_model_type,
            group_col=_group_col_str,
            use_temporal_cv=_tcv_str == "temporal",
        )
        _loaded.append(f"{_model_type}/{_ablation}/{_group_col_str}/{_tcv_str}")

    mo.stop(
        not _loaded,
        mo.callout(
            mo.md("No cached results found in `./cache/`. Run `./train-all.sh` first."),
            kind="warn",
        ),
    )

    perf_df = pl.read_csv(_csv)

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
    _MODEL_LABELS = {
        "gbdt": "GBDT",
        "elasticnet": "ElasticNet",
        "lmm": "Linear Mixed Effects",
    }

    def build_paper_table(df: pl.DataFrame) -> pl.DataFrame:
        """Format a filtered performance DataFrame into a paper-ready R² table."""
        _r2_rows = df.filter(pl.col("split") == "test_r2").select(
            "model", "ablation", "spruce", "pine", "oak", "beech"
        )
        # Weighted mean R² = Σ_species (n_species/n_total × R²_species)
        _weighted = df.filter(pl.col("split") == "test_weight_r2").select(
            "model",
            "ablation",
            weighted_r2=(
                pl.col("spruce").cast(pl.Float64, strict=False).fill_null(0.0)
                + pl.col("pine").cast(pl.Float64, strict=False).fill_null(0.0)
                + pl.col("oak").cast(pl.Float64, strict=False).fill_null(0.0)
                + pl.col("beech").cast(pl.Float64, strict=False).fill_null(0.0)
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
                _model_ord=pl.when(pl.col("model") == "gbdt")
                .then(0)
                .when(pl.col("model") == "elasticnet")
                .then(1)
                .otherwise(2),
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

    mo.vstack(
        [
            mo.md(
                "**Table 2**: R² test scores on 5-fold cross-validation grouped by tree identifiers "
                + ("with" if temporal_cv == "yes" else "without")
                + " temporal blocking"
            ),
            mo.ui.table(
                build_paper_table(
                    perf_df.filter(pl.col("group_by") == "tree_id").filter(
                        pl.col("temporal_cv") == temporal_cv
                    )
                ),
                page_size=20,
            ),
        ]
        for temporal_cv in ["yes", "no"]
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


@app.cell
def _(build_paper_table, mo, perf_df, pl):
    _df_plot = perf_df.filter(
        (pl.col("group_by") == "tree_id") & (pl.col("ablation") == "all")
    ).filter(pl.col("temporal_cv") == "no")

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
    model_type_ui = mo.ui.dropdown(
        ["gbdt", "elasticnet", "lmm"], value="gbdt", label="Model"
    )
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
    return (
        ablation_ui,
        group_col_ui,
        model_type_ui,
        temporal_cv_ui,
        weight_shap_ui,
    )


@app.cell
def _(
    ablation_ui,
    group_col_ui,
    joblib,
    mo,
    model_type_ui,
    os,
    temporal_cv_ui,
    weight_shap_ui,
):
    model_type = model_type_ui.value
    ablation = ablation_ui.value
    group_col = None if group_col_ui.value == "none" else group_col_ui.value
    use_temporal_cv = temporal_cv_ui.value
    weight_shap_fimp = weight_shap_ui.value

    _tcv = "temporal" if use_temporal_cv else "standard"
    _results_path = f"./cache/results-{ablation}-{model_type}-{group_col}-{_tcv}.pkl"
    _tcv_flag = "--temporal-cv" if use_temporal_cv else ""
    mo.stop(
        not os.path.exists(_results_path),
        mo.callout(
            mo.md(
                f"No cached results at `{_results_path}`.\n\nRun `./train-all.sh` (or "
                f"`uv run train.py --model-type {model_type} --ablation {ablation} "
                f"--group-col {group_col or 'none'} {_tcv_flag}`) first."
            ),
            kind="warn",
        ),
    )
    all_results = joblib.load(_results_path)
    mo.md(f"Loaded **{len(all_results)} species** from `{_results_path}`")
    return (
        ablation,
        all_results,
        group_col,
        model_type,
        use_temporal_cv,
        weight_shap_fimp,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Hyperparameters
    """)
    return


@app.cell
def _(ablation, group_col, mo, model_type, os, pl, use_temporal_cv):
    _tcv = "temporal" if use_temporal_cv else "standard"
    _hp_path = f"./cache/hyperparams-{ablation}-{model_type}-{group_col}-{_tcv}.parquet"

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
        .with_columns(
            pl.col("species").cast(pl.Enum(["spruce", "pine", "oak", "beech"]))
        )
        .sort("species")
        .with_columns(pl.col("species").cast(pl.Utf8))
        .transpose(column_names="species", include_header=True, header_name="Parameter")
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


@app.cell(hide_code=True)
def _(mo, model_type):
    mo.stop(
        model_type not in ("elasticnet", "lmm"),
        mo.callout(
            mo.md(
                "Preprocessing drop report is only available for linear models (ElasticNet, LMM)."
            ),
            kind="info",
        ),
    )
    mo.md("### Features dropped by preprocessing")
    return


@app.cell
def _(all_results, cs, mo, model_type, pl):
    from models import ElasticNetEstimator, MixedLMEstimator as _MixedLMEstimator

    mo.stop(model_type not in ("elasticnet", "lmm"))

    _rows = []
    for _species, _res in all_results.items():
        _feat_names = _res.features
        for _fold, _est in enumerate(_res.estimators):
            if not isinstance(_est, (ElasticNetEstimator, _MixedLMEstimator)):
                continue
            _miss_mask = (
                _est._miss_filter.get_support()
            )  # True = kept after missingness filter
            _var_mask = _est._var_mask  # True = kept after both filters
            for _i, _feat in enumerate(_feat_names):
                if not _miss_mask[_i]:
                    _rows.append(
                        {
                            "species": _species,
                            "fold": _fold,
                            "feature": _feat,
                            "reason": "missingness >50%",
                        }
                    )
                elif not _var_mask[_i]:
                    _rows.append(
                        {
                            "species": _species,
                            "fold": _fold,
                            "feature": _feat,
                            "reason": "near-zero variance",
                        }
                    )

    mo.stop(
        not _rows,
        mo.callout(
            mo.md("No features were dropped by the preprocessing pipeline."),
            kind="success",
        ),
    )

    _dropped = pl.DataFrame(_rows)
    num_folds = next(iter(all_results.values())).num_folds

    # Aggregate: one row per (species, feature, reason), with fold count and list
    dropped_features = (
        _dropped.group_by("species", "feature", "reason")
        .agg(
            n_folds=pl.col("fold").n_unique(),
            folds=pl.col("fold").sort().cast(pl.Utf8).implode().list.join(", "),
        )
        .sort("species", "reason", "feature")
    )

    # Cross-species summary: features dropped in every species × fold combination
    _consistent = (
        _dropped.group_by("feature", "reason", "species")
        .agg(n_folds=pl.col("fold").n_unique())
        .filter(pl.col("n_folds") > 0)
        .sort("reason", "feature")
        .pivot(on="species", values="n_folds")
        .pipe(
            lambda df: df.select(
                pl.col("feature"),
                pl.col("reason"),
                *[
                    pl.col(s) / num_folds * 100
                    for s in ["spruce", "pine", "oak", "beech"]
                    if s in df.columns
                ],
                pl.mean_horizontal(cs.numeric().fill_null(0) / num_folds * 100).alias(
                    "total [%]"
                ),
            )
        )
        .sort(by=cs.contains("total"), descending=True)
    )

    mo.vstack(
        [
            mo.md(
                f"Features dropped in at least one CV fold (out of {num_folds}) "
                "by the missingness filter (>50% missing) or near-zero variance filter."
            ),
            mo.ui.table(dropped_features),
            mo.md(
                f"**Consistently dropped** across all species and folds "
                f"({len(_consistent)} feature(s)):"
            ),
            mo.ui.table(_consistent) if len(_consistent) else mo.md("_None_"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, model_type):
    mo.stop(
        model_type != "lmm",
        mo.callout(
            mo.md(
                "Variance components and ICC are only available for the **LMM** model."
            ),
            kind="info",
        ),
    )
    mo.md("### Variance components (LMM)")
    return


@app.cell
def _(
    ablation,
    all_results,
    group_col,
    mo,
    model_type,
    np,
    os,
    pl,
    use_temporal_cv,
):
    mo.stop(model_type != "lmm")

    _tcv = "temporal" if use_temporal_cv else "standard"
    _hp_path = f"./cache/hyperparams-{ablation}-{model_type}-{group_col}-{_tcv}.parquet"
    mo.stop(
        not os.path.exists(_hp_path),
        mo.callout(
            mo.md(f"`{_hp_path}` not found — re-run training first."), kind="warn"
        ),
    )

    _hp = pl.read_parquet(_hp_path)

    # Per-fold R² / RMSE from already-loaded all_results
    _perf_fold = pl.DataFrame(
        [
            {
                "species": sp,
                "fold": fold,
                "R²": float(f["test_r2"]),
                "RMSE": float(f["test_rmse"]),
            }
            for sp, res in all_results.items()
            for fold, f in enumerate(res.performances)
        ]
    )

    # Per-fold table: variance components + performance
    _per_fold = (
        _hp.select("species", "fold", "var_random", "var_resid", "icc", "converged")
        .sort("species", "fold")
        .join(_perf_fold, on=["species", "fold"], how="left")
    )

    # Summary table: mean ± std across folds for ICC, R², RMSE
    _perf_summary = (
        pl.DataFrame(
            [
                {
                    "species": sp,
                    "r2_mean": float(np.mean([f["test_r2"] for f in res.performances])),
                    "r2_std": float(np.std([f["test_r2"] for f in res.performances])),
                    "rmse_mean": float(
                        np.mean([f["test_rmse"] for f in res.performances])
                    ),
                    "rmse_std": float(
                        np.std([f["test_rmse"] for f in res.performances])
                    ),
                }
                for sp, res in all_results.items()
            ]
        )
        .with_columns(
            R2=pl.concat_str(
                pl.col("r2_mean").round(3).cast(pl.Utf8),
                pl.lit(" ± "),
                pl.col("r2_std").round(3).cast(pl.Utf8),
            ),
            RMSE=pl.concat_str(
                pl.col("rmse_mean").round(3).cast(pl.Utf8),
                pl.lit(" ± "),
                pl.col("rmse_std").round(3).cast(pl.Utf8),
            ),
        )
        .select("species", "R2", "RMSE")
    )

    icc_summary = (
        _hp.group_by("species")
        .agg(
            var_random=pl.col("var_random").mean().round(4),
            var_resid=pl.col("var_resid").mean().round(4),
            icc=pl.concat_str(
                pl.col("icc").mean().round(3).cast(pl.Utf8),
                pl.lit(" ± "),
                pl.col("icc").std().round(3).cast(pl.Utf8),
            ),
            converged=pl.col("converged").all(),
        )
        .rename(
            {
                "var_random": "σ²_u (between-plot)",
                "var_resid": "σ²_e (within-plot)",
                "icc": "ICC (mean ± std)",
            }
        )
        .join(_perf_summary, on="species", how="left")
        .with_columns(
            pl.col("species").cast(pl.Enum(["spruce", "pine", "oak", "beech"]))
        )
        .sort("species")
        .with_columns(pl.col("species").cast(pl.Utf8))
        .transpose(column_names="species", include_header=True, header_name="Metric")
    )

    mo.vstack(
        [
            mo.md(
                "**Intraclass correlation coefficient** ICC = σ²_u / (σ²_u + σ²_e): "
                "proportion of total variance attributable to between-plot differences. "
                "Mean ± std across 5 CV folds."
            ),
            mo.ui.table(icc_summary),
            mo.md("**Per-fold values**"),
            mo.ui.table(_per_fold),
        ]
    )
    return


@app.cell
def _(
    ablation,
    all_results,
    cs,
    group_col,
    model_type,
    np,
    pl,
    use_temporal_cv,
):
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

    _tcv = "temporal" if use_temporal_cv else "standard"
    feature_importances.write_parquet(
        f"./cache/feature_importances-{ablation}-{model_type}-{group_col}-{_tcv}.parquet"
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
        palette=sns.color_palette("plasma", n_colors=_n_species + 1),
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
    _model_label = {"gbdt": "GBDT", "elasticnet": "ElasticNet", "lmm": "LMM"}.get(
        model_type, model_type
    )
    plt.title(
        f"Feature importance ({_model_label}, "
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
def _(
    ablation,
    feature_importances,
    group_col,
    mo,
    model_type,
    os,
    pl,
    use_temporal_cv,
):
    _other = "no-defoliation" if ablation == "all" else "all"
    _tcv = "temporal" if use_temporal_cv else "standard"
    _other_path = (
        f"./cache/feature_importances-{_other}-{model_type}-{group_col}-{_tcv}.parquet"
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
    dep_original_space_ui = mo.ui.switch(value=False, label="Original space")
    mo.hstack([dep_feature_ui, dep_species_ui, dep_original_space_ui], gap=2)
    return dep_feature_ui, dep_original_space_ui, dep_species_ui


@app.cell
def _(
    ALL_SPECIES,
    all_results,
    dep_feature_ui,
    dep_original_space_ui,
    dep_species_ui,
    plot_dependence,
    plt,
):
    _feature = dep_feature_ui.value
    _species_sel = dep_species_ui.value
    _use_original = dep_original_space_ui.value
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
            use_original_space=_use_original,
            ax=_ax,
        )
        _ax.set_title(_sp.capitalize())
        _ax.set_xlabel(_feature)
        _ax.set_ylabel(
            "SHAP value" if _use_original else "SHAP value [percentile rank %]"
        )

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


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Defoliation analysis
    """)
    return


@app.cell
def _(ALL_SPECIES, PlotType, all_results, plot_dependence, plt):
    _fig, _axes = plt.subplots(2, 2, figsize=(12, 8))

    for _sp, _ax in zip(ALL_SPECIES, _axes.flatten()):
        plot_dependence(
            all_results[_sp],
            feature="defoliation_mean",
            ax=_ax,
            xlim=(0, 100),
            plot_type=PlotType.LINE,
            linewidth=4.0,
            show_no_effect=False,
            ylim=(-15, 10),
        )
        plot_dependence(
            all_results[_sp],
            feature="defoliation_max",
            ax=_ax,
            xlim=(0, 100),
            color="#ff7f0e",
            plot_type=PlotType.LINE,
            linewidth=4.0,
            show_no_effect=False,
            ylim=(-15, 10),
            with_density=False,
        )
        _ax.set_title(_sp.capitalize())
        _ax.set_xlabel("Defoliation [%]")
        _ax.set_ylabel("SHAP value [percentile rank %]")

    _fig.legend(
        title="Feature",
        labels=[
            "Mean defoliation (μ)",
            "Mean defoliation (95p)",
            "Max defoliation (μ)",
            "Max defoliation (95p)",
        ],
    )
    _fig.suptitle("Dependence of growth rate on mean and max defoliation")
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(ALL_SPECIES, all_results, np, plt):
    import matplotlib.cm as _cm
    from matplotlib.colors import TwoSlopeNorm as _TwoSlopeNorm

    _norm = _TwoSlopeNorm(vmin=-0.30, vcenter=0, vmax=0.2)
    _cmap = plt.get_cmap("coolwarm")
    _fold = 0

    _fig, _axes = plt.subplots(2, 2, figsize=(12, 9))

    for _sp, _ax in zip(ALL_SPECIES, _axes.flatten()):
        _data = all_results[_sp].get_data(fold=_fold, split="all")[0]
        _shap_vals = (
            all_results[_sp].shap_values[_fold][:, "defoliation_median"].values
            + all_results[_sp].shap_values[_fold][:, "defoliation_max"].values
        )

        _ax.plot([0, 100], [0, 100], color="grey", linestyle="--", linewidth=1)
        _ax.text(5, 5, "y = x", color="grey", fontsize=10, ha="left", va="top")
        _ax.scatter(
            _data["defoliation_median"].to_numpy(),
            _data["defoliation_max"].to_numpy(),
            c=_shap_vals,
            alpha=0.8,
            s=15,
            cmap=_cmap,
            norm=_norm,
        )

        _r2 = (
            np.corrcoef(
                _data["defoliation_median"].to_numpy(),
                _data["defoliation_max"].to_numpy(),
            )[0, 1]
            ** 2
        )
        _ax.set_title(f"{_sp.capitalize()} (R² = {_r2:.2f})", fontsize=12)
        _ax.set_xlabel("Median defoliation")
        _ax.set_ylabel("Max defoliation")
        _ax.set_xlim(0, 100)
        _ax.set_ylim(0, 100)

    _fig.colorbar(
        _cm.ScalarMappable(norm=_norm, cmap=_cmap),
        ax=_axes,
        cax=_fig.add_axes([1.01, 0.15, 0.02, 0.7]),
        label="Sum of SHAP values of median and max defoliation",
        orientation="vertical",
        use_gridspec=True,
    )
    _fig.suptitle(
        f"Relation between median and max defoliation (fold {_fold})", fontsize=16
    )
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
