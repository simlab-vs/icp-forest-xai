import os
import numpy as np
import polars as pl
import polars.selectors as cs

from models import ExperimentResults, Species, ModelType, ALL_SPECIES
from config import Ablation


def check_significance(r2: str, k: int = 5) -> str:
    """Check if the R2 value is significant."""
    from scipy import stats

    # Split the R2 value of the form "0.307 ± 0.02"
    r2_value, r2_error = r2.split(" ± ")

    # Perform a t-test to check if the R2 value is significantly positive
    standard_error = float(r2_error) / np.sqrt(k)
    t_statistic = float(r2_value) / standard_error

    # Calculate the p-value
    p_value = stats.t.sf(t_statistic, k - 1)

    if p_value < 0.001:
        return f"{r2_value} ± {r2_error}***"
    elif p_value < 0.01:
        return f"{r2_value} ± {r2_error}**"
    elif p_value < 0.05:
        return f"{r2_value} ± {r2_error}*"
    else:
        return f"{r2_value} ± {r2_error}"


PERF_CSV = "./cache/performance_summary.csv"
PERF_KEYS = ["group_by", "model", "split", "ablation"]


def summarize_performance(
    all_results: dict[Species, ExperimentResults],
    ablation: Ablation,
    model_type: ModelType,
    group_col: str,
    precision: int = 2,
) -> None:
    perf = pl.concat(
        [
            pl.from_dicts(results.performances).select(
                pl.lit(species).alias("species"),
                pl.first().cum_count().alias("fold"),
                "test_r2",
                "train_r2",
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
        )
        .select(
            "species",
            test=pl.col("mean_test_r2").round(precision).cast(pl.Utf8)
            + " ± "
            + pl.col("std_test_r2").round(precision).cast(pl.Utf8),
            train=pl.col("mean_train_r2").round(precision).cast(pl.Utf8)
            + " ± "
            + pl.col("std_train_r2").round(precision).cast(pl.Utf8),
        )
        .unpivot(index="species")
        .pivot(index="variable", on="species", values="value")
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
        cs.by_dtype(pl.Utf8).str.replace_all("+/-", "±", literal=True).name.keep()
    )

    perf.write_csv(PERF_CSV)

    with pl.Config() as cfg:
        cfg.set_tbl_formatting("ASCII_MARKDOWN")
        cfg.set_tbl_width_chars(125)
        cfg.set_tbl_hide_column_data_types(True)

        for group_by in perf["group_by"].unique().sort():
            print(f"\nPerformance summary for group_by='{group_by}':")
            print(
                perf.filter(pl.col("group_by") == group_by)
                .filter(pl.col("split") != "train")
                .with_columns(
                    mean_r2=pl.mean_horizontal(
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
                .select(cs.all().exclude("group_by", "split"))
            )
