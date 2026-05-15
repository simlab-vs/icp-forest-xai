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
    # Supplementary XAI Analyses

    Loads pre-computed results from `./cache/` (run `./train-all.sh` first).
    Select experiment configuration below.
    """)
    return


@app.cell
def _():
    import joblib
    import numpy as np
    import shap
    import polars as pl
    import polars.selectors as cs
    import matplotlib.pyplot as plt
    import seaborn as sns
    import networkx as nx
    import os
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE

    from models import ALL_SPECIES
    from config import FEATURES_METADATA
    from explain import (
        compute_interaction_matrix,
        plot_ceteris_paribus_profile,
    )

    return (
        ALL_SPECIES,
        FEATURES_METADATA,
        StandardScaler,
        TSNE,
        compute_interaction_matrix,
        cs,
        joblib,
        np,
        nx,
        os,
        pl,
        plot_ceteris_paribus_profile,
        plt,
        shap,
        sns,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Performance Tables (RMSE)

    RMSE counterparts to Tables 2–4 from the main analysis.
    Use the selectors to switch between tree-level / plot-level grouping and temporal vs. standard cross-validation.
    """)
    return


@app.cell
def _(mo, os, pl):
    _csv = "./cache/performance_summary.csv"
    mo.stop(
        not os.path.exists(_csv),
        mo.callout(
            mo.md(
                "Performance summary not found. "
                "Open and run `02-xai-analysis.py` first to generate "
                "`./cache/performance_summary.csv`."
            ),
            kind="warn",
        ),
    )
    perf_df = pl.read_csv(_csv)
    return (perf_df,)


@app.cell
def _(mo):
    rmse_group_col_ui = mo.ui.dropdown(
        ["tree_id", "plot_id"], value="tree_id", label="Group by"
    )
    rmse_temporal_cv_ui = mo.ui.switch(value=False, label="Temporal CV")
    mo.hstack([rmse_group_col_ui, rmse_temporal_cv_ui], gap=2)
    return rmse_group_col_ui, rmse_temporal_cv_ui


@app.cell
def _(mo, perf_df, pl, rmse_group_col_ui, rmse_temporal_cv_ui):
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

    def build_rmse_table(df: pl.DataFrame) -> pl.DataFrame:
        _rmse_rows = df.filter(pl.col("split") == "test_rmse").select(
            "model", "ablation", "spruce", "pine", "oak", "beech"
        )
        _weighted = df.filter(pl.col("split") == "test_weight_rmse").select(
            "model",
            "ablation",
            weighted_rmse=(
                pl.col("spruce").cast(pl.Float64, strict=False).fill_null(0.0)
                + pl.col("pine").cast(pl.Float64, strict=False).fill_null(0.0)
                + pl.col("oak").cast(pl.Float64, strict=False).fill_null(0.0)
                + pl.col("beech").cast(pl.Float64, strict=False).fill_null(0.0)
            )
            .round(2)
            .cast(pl.Utf8),
        )
        return (
            _rmse_rows.join(_weighted, on=["model", "ablation"], how="left")
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
                    "weighted_rmse": "Weighted RMSE",
                }
            )
            .select("config", "Spruce", "Pine", "Beech", "Oak", "Weighted RMSE")
            .rename({"config": "Configuration"})
        )

    _group_col = rmse_group_col_ui.value
    _tcv = "yes" if rmse_temporal_cv_ui.value else "no"
    _is_plot = _group_col == "plot_id"

    _df_sel = perf_df.filter(
        (pl.col("group_by") == _group_col) & (pl.col("temporal_cv") == _tcv)
    )
    if _is_plot:
        _df_sel = _df_sel.filter(pl.col("ablation") == "all")

    mo.stop(
        _df_sel.is_empty(),
        mo.callout(
            mo.md(
                f"No data for group_by=`{_group_col}`, temporal_cv=`{_tcv}`. "
                "Run `./train-all.sh` first."
            ),
            kind="info",
        ),
    )

    _tcv_label = "with" if _tcv == "yes" else "without"
    _group_label = "plot identifiers" if _is_plot else "tree identifiers"
    if _is_plot:
        _table_num = "Table 4 (RMSE)"
        _caption = (
            f"**{_table_num}**: RMSE test scores on 5-fold cross-validation grouped by "
            "plot identifiers. Best strictly positive score for each species is indicated in bold."
        )
    else:
        _table_num = "Table 2 (RMSE)" if _tcv == "no" else "Table 3 (RMSE)"
        _caption = (
            f"**{_table_num}**: RMSE test scores on 5-fold cross-validation grouped by "
            f"tree identifiers {_tcv_label} temporal blocking for different ablation studies "
            "on GBDT models and linear models, with different feature sets."
        )

    mo.vstack(
        [
            mo.md(_caption),
            mo.ui.table(build_rmse_table(_df_sel), page_size=20),
        ]
    )
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
    temporal_cv_ui = mo.ui.switch(value=False, label="Temporal CV")
    mo.hstack([model_type_ui, ablation_ui, group_col_ui, temporal_cv_ui], gap=2)
    return ablation_ui, group_col_ui, model_type_ui, temporal_cv_ui


@app.cell
def _(
    ablation_ui,
    cs,
    group_col_ui,
    joblib,
    mo,
    model_type_ui,
    np,
    os,
    pl,
    temporal_cv_ui,
):
    model_type = model_type_ui.value
    ablation = ablation_ui.value
    group_col = None if group_col_ui.value == "none" else group_col_ui.value
    use_temporal_cv = temporal_cv_ui.value
    _tcv = "temporal" if use_temporal_cv else "standard"

    _results_path = f"./cache/results-{ablation}-{model_type}-{group_col}-{_tcv}.pkl"
    mo.stop(
        not os.path.exists(_results_path),
        mo.callout(
            mo.md(
                f"No cached results at `{_results_path}`.\n\nRun `./train-all.sh` first."
            ),
            kind="warn",
        ),
    )
    all_results = joblib.load(_results_path)
    _last_res = next(reversed(all_results.values()))

    feature_importances = pl.from_dicts(
        [
            {
                "species": sp,
                "fold": fold,
                **dict(
                    zip(
                        res.features,
                        np.abs(res.shap_values[fold].values).mean(axis=0),
                    )
                ),
                "n": len(res.shap_values[fold].values),
            }
            for sp, res in all_results.items()
            for fold in range(res.num_folds)
        ]
    ).unpivot(
        on=cs.exclude("species", "fold", "n"),
        index=["species", "fold", "n"],
        variable_name="feature",
        value_name="shap",
    )

    n_features = len(_last_res.features)

    mo.md(f"Loaded **{len(all_results)} species** from `{_results_path}`")
    return (
        ablation,
        all_results,
        feature_importances,
        group_col,
        model_type,
        n_features,
        use_temporal_cv,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Beeswarm plots
    """)
    return


@app.cell(hide_code=True)
def _(ALL_SPECIES, mo):
    beeswarm_species_ui = mo.ui.dropdown(
        ALL_SPECIES + ["all"], value="spruce", label="Species"
    )
    beeswarm_species_ui
    return (beeswarm_species_ui,)


@app.cell
def _(
    FEATURES_METADATA,
    all_results,
    beeswarm_species_ui,
    feature_importances,
    np,
    pl,
    plt,
    shap,
):
    _feature_to_label = {k: v["label"] for k, v in FEATURES_METADATA.items()}
    _sp_sel = beeswarm_species_ui.value
    _species_list = list(all_results.keys()) if _sp_sel == "all" else [_sp_sel]

    _features_ordered = (
        feature_importances.group_by("feature")
        .agg(pl.col("shap").mean().alias("importance"))
        .sort("importance", descending=True)["feature"]
        .map_elements(lambda f: _feature_to_label.get(f, f), return_dtype=pl.String)
        .to_list()
    )

    def _make_exp(results, features):
        vals = np.vstack([sv.values for sv in results.shap_values])
        data = np.vstack([sv.data for sv in results.shap_values])
        f2i = {_feature_to_label.get(f, f): i for i, f in enumerate(results.features)}
        idx = [f2i[f] for f in features if f in f2i]
        sel = [f for f in features if f in f2i]
        return shap.Explanation(
            values=vals[:, idx] * 100, data=data[:, idx], feature_names=sel
        )

    _n = len(_species_list)
    _row_h = max(6, len(_features_ordered) * 0.35)
    _fig, _axes = plt.subplots(_n, 1, figsize=(12, _row_h * _n), squeeze=False)

    for _i, _s in enumerate(_species_list):
        _exp = _make_exp(all_results[_s], _features_ordered)
        plt.sca(_axes[_i, 0])
        shap.plots.beeswarm(
            _exp,
            show=False,
            max_display=len(_features_ordered),
            color_bar=True,
            plot_size=None,
        )
        _axes[_i, 0].set_title(_s.capitalize(), fontsize=13)
        _axes[_i, 0].set_xlabel("SHAP value [quantile rank %]")

    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Ablation comparison
    """)
    return


@app.cell
def _(
    FEATURES_METADATA,
    ablation,
    all_results,
    cs,
    feature_importances,
    group_col,
    mo,
    model_type,
    os,
    pl,
    plt,
    sns,
    use_temporal_cv,
):
    _other = "no-defoliation" if ablation == "all" else "all"
    _tcv = "temporal" if use_temporal_cv else "standard"
    _other_path = (
        f"./cache/feature_importances-{_other}-{model_type}-{group_col}-{_tcv}.parquet"
    )
    _this_path = f"./cache/feature_importances-{ablation}-{model_type}-{group_col}-{_tcv}.parquet"

    mo.stop(
        not os.path.exists(_other_path) or not os.path.exists(_this_path),
        mo.callout(
            mo.md(f"Comparison skipped — `{_other_path}` not found."), kind="info"
        ),
    )

    _f2l = {k: v["label"] for k, v in FEATURES_METADATA.items()}
    _top_n = 10

    _comp = (
        feature_importances.group_by("species", "feature")
        .agg(pl.col("shap").mean().alias(f"shap-{ablation}") * 100)
        .join(
            pl.read_parquet(_other_path)
            .group_by("species", "feature")
            .agg(pl.col("shap").mean().alias(f"shap-{_other}") * 100),
            on=["species", "feature"],
            how="full",
            validate="1:1",
            coalesce=True,
        )
    )

    _n_sp = len(all_results)
    _fig, _axes = plt.subplots(1, _n_sp, figsize=(5 * _n_sp, 7), squeeze=False)

    for _sp, _ax in zip(all_results.keys(), _axes.flatten()):
        _data = (
            _comp.filter(pl.col("species") == _sp)
            .with_columns(
                feature=pl.col("feature").map_elements(
                    lambda f: _f2l.get(f, f), return_dtype=pl.String
                )
            )
            .group_by("feature")
            .agg(
                pl.col(f"shap-{ablation}").mean().alias(f"imp-{ablation}"),
                pl.col(f"shap-{_other}").mean().alias(f"imp-{_other}"),
            )
            .with_columns(
                pl.max_horizontal(pl.col(f"imp-{ablation}"), pl.col(f"imp-{_other}"))
                .rank(descending=True)
                .alias("rank")
            )
        )
        _data = (
            _data.unpivot(index=["feature", "rank"], on=cs.starts_with("imp-"))
            .with_columns(ablation_label=pl.col("variable").str.replace("imp-", ""))
            .select(
                "feature",
                "ablation_label",
                pl.col("value").alias("importance"),
                "rank",
            )
        )
        _data = pl.concat(
            [
                _data.filter(pl.col("rank") <= _top_n)
                .sort("rank")
                .select("feature", "ablation_label", "importance"),
                _data.filter(pl.col("rank") > _top_n).select(
                    pl.lit("all other").alias("feature"),
                    "ablation_label",
                    pl.col("importance")
                    .sum()
                    .over("ablation_label")
                    .alias("importance"),
                ),
            ]
        )
        sns.barplot(
            data=_data,
            x="importance",
            y="feature",
            hue="ablation_label",
            palette=sns.color_palette("cmo.thermal", n_colors=2),
            ax=_ax,
        )
        for _c in _ax.containers:
            _ax.bar_label(_c, fontsize=9, fmt="%.1f%%", padding=2)
        _ax.set_xlim(_ax.get_xlim()[0], _ax.get_xlim()[1] * 1.2)
        _ax.set_xlabel("Mean |SHAP| %")
        _ax.set_ylabel("Feature")
        _ax.set_title(_sp.capitalize())

    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Ceteris paribus profiles
    """)
    return


@app.cell(hide_code=True)
def _(ALL_SPECIES, FEATURES_METADATA, mo):
    cp_species_ui = mo.ui.dropdown(ALL_SPECIES, value="spruce", label="Species")
    cp_feature_ui = mo.ui.dropdown(
        list(FEATURES_METADATA.keys()), value="defoliation_mean", label="Feature"
    )
    cp_fold_ui = mo.ui.slider(0, 4, value=0, label="Fold")
    mo.hstack([cp_species_ui, cp_feature_ui, cp_fold_ui], gap=2)
    return cp_feature_ui, cp_fold_ui, cp_species_ui


@app.cell
def _(
    all_results,
    cp_feature_ui,
    cp_fold_ui,
    cp_species_ui,
    np,
    plot_ceteris_paribus_profile,
    plt,
):
    _sp = cp_species_ui.value
    _feature = cp_feature_ui.value
    _fold = cp_fold_ui.value

    _res = all_results[_sp]
    _X, _, _y_pred = _res.get_data(_fold, "test")
    _y_vec = _y_pred.to_numpy()

    _fig, _axes = plt.subplots(2, 2, figsize=(10, 8))
    _bands = [(0, 5), (20, 35), (70, 80), (95, 100)]

    for (_lo_p, _hi_p), _ax in zip(_bands, _axes.flat):
        _lo = np.min(_y_vec) if _lo_p == 0 else np.percentile(_y_vec, _lo_p)
        _hi = np.max(_y_vec) if _hi_p == 100 else np.percentile(_y_vec, _hi_p)
        _idxs = np.argwhere((_y_vec >= _lo) & (_y_vec < _hi)).flatten()
        if len(_idxs) > 0:
            for _i in np.random.choice(_idxs, min(5, len(_idxs)), replace=False):
                plot_ceteris_paribus_profile(
                    _res.estimators[_fold], _X, _i, _feature, ax=_ax
                )
        _ax.set_title(f"Growth rate [{_lo:.2f}, {_hi:.2f}]")

    _fig.suptitle(f"Ceteris paribus — {_sp.capitalize()} / {_feature}")
    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## SHAP vector clustering

    Intra- vs. inter-cluster pairwise distances of SHAP vectors grouped by `plot_id`.
    """)
    return


@app.cell
def _(all_results, cs, np, pl):
    _shap_list, _sp_col, _plot_col, _tree_col = [], [], [], []
    _last = next(reversed(all_results.values()))

    for _sp, _res in all_results.items():
        for _fold in range(_res.num_folds):
            _arr = _res.shap_values[_fold].values
            _shap_list.append(_arr)
            _sp_col.extend([_sp] * len(_arr))
            _plot_col.extend(_res.metadata["plot_id"].to_numpy())
            _tree_col.extend(_res.metadata["tree_id"].to_numpy())

    df_shap = (
        pl.from_numpy(np.concatenate(_shap_list, axis=0), schema=_last.features)
        .with_columns(
            pl.Series("species", _sp_col),
            pl.Series("plot_id", _plot_col),
            pl.Series("tree_id", _tree_col),
        )
        .select(
            "species",
            "plot_id",
            "tree_id",
            pl.concat_arr(cs.exclude("species", "plot_id", "tree_id")).alias("shap"),
        )
        .sample(n=2000)
    )
    return (df_shap,)


@app.cell
def _(df_shap, n_features, pl):
    _D = n_features
    _comp_cols = [pl.col("shap").arr.get(i).alias(f"c{i}") for i in range(_D)]
    _dfc = df_shap.with_columns(_comp_cols).with_columns(
        sq_norm=sum(pl.col(f"c{i}") ** 2 for i in range(_D))
    )

    _glob = (
        _dfc.select(
            n=pl.len(),
            S2=pl.col("sq_norm").sum(),
            **{f"S1_{i}": pl.col(f"c{i}").sum() for i in range(_D)},
        )
        .with_columns(S1_sq=sum(pl.col(f"S1_{i}") ** 2 for i in range(_D)))
        .select("n", "S2", "S1_sq")
    )

    _per = (
        _dfc.group_by("plot_id")
        .agg(
            n_k=pl.len(),
            S2_k=pl.col("sq_norm").sum(),
            **{f"S1k_{i}": pl.col(f"c{i}").sum() for i in range(_D)},
        )
        .with_columns(S1k_sq=sum(pl.col(f"S1k_{i}") ** 2 for i in range(_D)))
        .with_columns(
            T_k=pl.col("n_k") * pl.col("S2_k") - pl.col("S1k_sq"),
            P_k=(pl.col("n_k") * (pl.col("n_k") - 1)) / 2,
        )
        .select("plot_id", "n_k", "T_k", "P_k")
    )

    clustering_metrics = (
        _per.select(
            T_intra=pl.col("T_k").sum(),
            P_intra=pl.col("P_k").sum(),
            n=pl.lit(_glob.item(0, "n")),
            S2=pl.lit(_glob.item(0, "S2")),
            S1_sq=pl.lit(_glob.item(0, "S1_sq")),
        )
        .with_columns(
            P_all=(pl.col("n") * (pl.col("n") - 1)) / 2,
            T_all=pl.col("n") * pl.col("S2") - pl.col("S1_sq"),
        )
        .with_columns(
            T_inter=pl.col("T_all") - pl.col("T_intra"),
            P_inter=pl.col("P_all") - pl.col("P_intra"),
        )
        .with_columns(
            rms_intra=(pl.col("T_intra") / pl.col("P_intra")).sqrt(),
            rms_inter=(pl.col("T_inter") / pl.col("P_inter")).sqrt(),
        )
        .with_columns(clusterness=pl.col("rms_intra") / pl.col("rms_inter"))
        .select("rms_intra", "rms_inter", "clusterness")
    )

    clustering_metrics
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## t-SNE projection of SHAP vectors
    """)
    return


@app.cell
def _(StandardScaler, TSNE, df_shap, np):
    _scaler = StandardScaler()
    _X_scaled = np.nan_to_num(_scaler.fit_transform(df_shap["shap"].to_numpy()))
    _tsne = TSNE(n_components=2, perplexity=100, early_exaggeration=20)
    X_tsne = _tsne.fit_transform(_X_scaled)
    return (X_tsne,)


@app.cell
def _(X_tsne, df_shap, pl, plt, sns):
    _df_tsne = df_shap.with_columns(
        pl.Series("tsne_x", X_tsne[:, 0]),
        pl.Series("tsne_y", X_tsne[:, 1]),
    )

    _fig, _axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.scatterplot(
        data=_df_tsne.to_pandas(),
        x="tsne_x",
        y="tsne_y",
        hue="species",
        alpha=0.5,
        palette="muted",
        s=15,
        ax=_axes[0],
    )
    _axes[0].legend_.set_title("Species")
    for _lbl in _axes[0].legend_.texts:
        _lbl.set_text(_lbl.get_text().capitalize())
    _axes[0].set_xlabel("Dimension 1")
    _axes[0].set_ylabel("Dimension 2")
    _axes[0].set_title("By species")

    sns.scatterplot(
        data=_df_tsne.to_pandas(),
        x="tsne_x",
        y="tsne_y",
        hue="plot_id",
        alpha=0.5,
        palette="dark",
        legend=False,
        s=10,
        ax=_axes[1],
    )
    _axes[1].set_xlabel("Dimension 1")
    _axes[1].set_ylabel("Dimension 2")
    _axes[1].set_title("By plot_id")

    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Feature interaction network
    """)
    return


@app.cell(hide_code=True)
def _(ALL_SPECIES, mo):
    net_species_ui = mo.ui.dropdown(ALL_SPECIES, value="oak", label="Species")
    net_cutoff_ui = mo.ui.slider(
        0.0005, 0.01, value=0.002, step=0.0005, label="Edge cutoff"
    )
    mo.hstack([net_species_ui, net_cutoff_ui], gap=2)
    return net_cutoff_ui, net_species_ui


@app.cell
def _(
    all_results,
    compute_interaction_matrix,
    feature_importances,
    net_species_ui,
    pl,
    plt,
):
    _sp = net_species_ui.value
    _top_n = (
        feature_importances.select(
            "feature",
            pl.col("shap").mean().over("feature").alias("importance"),
        )
        .unique()
        .sort("importance", descending=True)
        .head(20)["feature"]
        .to_list()
    )

    plt.figure(figsize=(10, 8))
    interactions_matrix, _ = compute_interaction_matrix(
        all_results[_sp], top_n=_top_n, ax=plt.gca(), vmax=0.006
    )
    plt.title(f"Feature interactions — {_sp.capitalize()}")
    plt.tight_layout()
    plt.gca()
    return (interactions_matrix,)


@app.cell
def _(
    all_results,
    interactions_matrix,
    net_cutoff_ui,
    net_species_ui,
    np,
    nx,
    plt,
):
    _sp = net_species_ui.value
    _cutoff = net_cutoff_ui.value
    _res = all_results[_sp]

    _adj = np.triu(np.absolute(interactions_matrix).mean(axis=0), k=1)
    _adj[_adj < _cutoff] = 0.0

    _G = nx.from_numpy_array(_adj, edge_attr="interaction", nodelist=_res.features)
    _G.remove_nodes_from(list(nx.isolates(_G)))

    plt.figure(figsize=(12, 12))
    _pos = nx.circular_layout(_G)
    nx.draw_networkx_edges(
        _G, _pos, width=[_G[u][v]["interaction"] * 1000 for u, v in _G.edges()]
    )
    nx.draw_networkx_edge_labels(
        _G,
        _pos,
        edge_labels={
            k: f"{v:.2e}" for k, v in nx.get_edge_attributes(_G, "interaction").items()
        },
    )
    nx.draw_networkx_labels(
        _G,
        _pos,
        font_size=12,
        font_color="black",
        bbox=dict(facecolor="lightblue", boxstyle="round,pad=0.5,rounding_size=0.5"),
    )
    plt.title(f"Interaction graph — {_sp.capitalize()} (cutoff={_cutoff:.4f})")
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
