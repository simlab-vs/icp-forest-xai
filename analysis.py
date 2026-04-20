import os
import numpy as np
import polars as pl
import polars.selectors as cs

from models import ExperimentResults, Species, ModelType, ALL_SPECIES
from config import Ablation


def check_significance(metric: str, k: int = 5) -> str:
    """Check if the R2 value is significant."""
    from scipy import stats

    # Split the R2/rmse value of the form "0.307 ± 0.02"
    metric_value, metric_error = metric.split(" ± ")

    # Perform a t-test to check if the R2/rmse value is significantly positive
    standard_error = float(metric_error) / np.sqrt(k)
    t_statistic = float(metric_value) / standard_error

    # Calculate the p-value
    p_value = stats.t.sf(t_statistic, k - 1)

    if p_value < 0.001:
        return f"{metric_value} ± {metric_error}***"
    elif p_value < 0.01:
        return f"{metric_value} ± {metric_error}**"
    elif p_value < 0.05:
        return f"{metric_value} ± {metric_error}*"
    else:
        return f"{metric_value} ± {metric_error}"


PERF_CSV = "./cache/performance_summary.csv"
PERF_KEYS = ["group_by", "model", "split", "ablation"]


def summarize_performance(
    all_results: dict[Species, ExperimentResults],
    ablation: Ablation,
    model_type: ModelType,
    group_col: str,
    use_temporal_cv = False,
    precision: int = 2,
) -> None:
    perf = pl.concat(
        [
            pl.from_dicts(results.performances).select(
                pl.lit(species).alias("species"),
                pl.first().cum_count().alias("fold"),
                "test_r2",
                "train_r2",
                "test_rmse",
                "train_rmse",
                "n_train",
                "n_test",
            )
            for species, results in all_results.items()
        ],
        how="vertical",
    )

    perf = (
        perf.group_by("species")
        .agg(
            pl.mean("test_r2").alias("mean_test_r2"),
            pl.std("test_r2").alias("std_test_r2"),
            pl.mean("train_r2").alias("mean_train_r2"),
            pl.std("train_r2").alias("std_train_r2"),
            pl.mean("test_rmse").alias("mean_test_rmse"),
            pl.std("test_rmse").alias("std_test_rmse"),
            pl.mean("train_rmse").alias("mean_train_rmse"),
            pl.std("train_rmse").alias("std_train_rmse"),
            pl.mean("n_test").round().cast(pl.Int32).alias("n_test"),
            pl.mean("n_train").round().cast(pl.Int32).alias("n_train"),
        )
        .with_columns(
            weight_test=pl.col("n_test") / pl.sum("n_test").over(pl.lit(True)),
            weight_train=pl.col("n_train") / pl.sum("n_train").over(pl.lit(True)),
        )
        .with_columns(
            weight_test_r2=pl.col("weight_test") * pl.col("mean_test_r2"),
            weight_test_rmse=pl.col("weight_test") * pl.col("mean_test_rmse"),
            weight_train_r2=pl.col("weight_train") * pl.col("mean_train_r2"),
            weight_train_rmse=pl.col("weight_test") * pl.col("mean_train_rmse"),
        )
        .select(
            "species",
            test_r2=pl.col("mean_test_r2").round(precision).cast(pl.Utf8)
            + " ± "
            + pl.col("std_test_r2").round(precision).cast(pl.Utf8),
            train_r2=pl.col("mean_train_r2").round(precision).cast(pl.Utf8)
            + " ± "
            + pl.col("std_train_r2").round(precision).cast(pl.Utf8),
            test_rmse=pl.col("mean_test_rmse").round(precision).cast(pl.Utf8)
            + " ± "
            + pl.col("std_test_rmse").round(precision).cast(pl.Utf8),
            train_rmse=pl.col("mean_train_rmse").round(precision).cast(pl.Utf8)
            + " ± "
            + pl.col("std_train_rmse").round(precision).cast(pl.Utf8),
            weight=pl.col("weight_test").round(precision).cast(pl.Utf8),
            test_weight_r2=pl.col("weight_test_r2").round(precision).cast(pl.Utf8),
            test_weight_rmse=pl.col("weight_test_rmse").round(precision).cast(pl.Utf8),
            train_weight_r2=pl.col("weight_train_r2").round(precision).cast(pl.Utf8),
            train_weight_rmse=pl.col("weight_train_rmse")
            .round(precision)
            .cast(pl.Utf8),
        )
        .unpivot(index=["species"])
        .pivot(index=["variable"], on="species", values="value")
        .rename({"variable": "split"})
        .select(
            pl.lit(ablation).cast(pl.Utf8).alias("ablation"),
            pl.lit(model_type).cast(pl.Utf8).alias("model"),
            pl.lit(group_col).cast(pl.Utf8).alias("group_by"),
            "split",
            *ALL_SPECIES,
        )
    )

    if os.path.exists(PERF_CSV):
        perf = pl.concat([perf, pl.read_csv(PERF_CSV)], how="vertical").unique(
            subset=PERF_KEYS, keep="last"
        )

    # Sort the performance DataFrame for better readability
    perf = perf.sort(PERF_KEYS).select(
        cs.by_dtype(pl.Utf8).str.replace_all("+/-", "±", literal=True).name.keep(),
    )

    perf.write_csv(PERF_CSV)

    with pl.Config() as cfg:
        cfg.set_tbl_formatting("ASCII_MARKDOWN")
        cfg.set_tbl_width_chars(125)
        cfg.set_tbl_hide_column_data_types(True)

        for group_by in perf["group_by"].unique().sort():
            if use_temporal_cv:
                temporal_str = "with temporal blocking"
            else:
                temporal_str = "without temporal blocking"
            print(f"\nPerformance summary for group_by='{group_by}' {temporal_str}:")

            weighted = (
                perf.filter(pl.col("group_by") == group_by)
                .filter(pl.col("split").str.contains("weight"))
                .filter(pl.col("split").str.starts_with("test"))
                .with_columns(
                    weighted=pl.sum_horizontal(
                        cs.by_name("spruce", "pine", "beech", "oak").cast(pl.Float64)
                    ).round(2),
                )
                .select(cs.all().exclude("group_by"))
            )

            weighted = weighted.select(
                ["ablation", "model", "split", "weighted"]
            ).with_columns(pl.col("split").str.replace("_weight", "").alias("split"))

            print(
                perf.filter(pl.col("group_by") == group_by)
                .filter(pl.col("split").str.starts_with("test"))
                .filter(~pl.col("split").str.contains("weight"))
                .with_columns(
                    mean_metric=pl.mean_horizontal(
                        cs.by_name("spruce", "pine", "beech", "oak")
                        .str.split(" ± ")
                        .list.get(0)
                        .cast(pl.Float64)
                    ).round(2),
                )
                .with_columns(
                    cs.by_name("spruce", "pine", "beech", "oak").map_elements(
                        lambda x: check_significance(x, k=5), return_dtype=pl.Utf8
                    )
                )
                .select(cs.all().exclude("group_by"))
                .join(weighted, on=["ablation", "model", "split"], how="left")
            )
