import os

import polars as pl
import polars.selectors as cs

from config import DATA_PATH, FEATURES_METADATA
from data import SPECIES_MAPPING


def summarize_data(df: pl.DataFrame) -> None:
    if "tree_id" not in df.columns:
        df_with_tree_id = (
            df.join(
                df_growth_raw.select("plot_id", "tree_id", "specie").unique(),
                on="plot_id",
            )
            .with_columns(
                species=pl.col("specie").cast(pl.Utf8).replace(SPECIES_MAPPING)
            )
            .filter(pl.col("species").is_in(SPECIES_MAPPING.values()))
        )

        df = pl.concat(
            [
                df.select(
                    pl.lit(None).alias("species"),
                    pl.n_unique("plot_id").alias("# plots"),
                    pl.lit(None).cast(pl.UInt32()).alias("# trees"),
                    pl.len().alias("# observations"),
                ),
                df_with_tree_id.group_by("species").agg(
                    pl.n_unique("plot_id").alias("# plots"),
                    pl.n_unique("tree_id").alias("# trees"),
                    pl.len().alias("# observations"),
                ),
                df_with_tree_id.select(
                    pl.lit("*").alias("species"),
                    pl.n_unique("plot_id").alias("# plots"),
                    pl.n_unique("tree_id").alias("# trees"),
                    pl.len().alias("# observations"),
                ),
            ],
            how="vertical_relaxed",
        )
    else:
        df = df.with_columns(
            species=pl.col("specie").cast(pl.Utf8).replace(SPECIES_MAPPING)
        ).filter(pl.col("species").is_in(SPECIES_MAPPING.values()))

        df = pl.concat(
            [
                df.group_by("species").agg(
                    pl.n_unique("plot_id").alias("# plots"),
                    pl.n_unique("tree_id").alias("# trees"),
                    pl.len().alias("# observations"),
                ),
                df.select(
                    pl.lit("*").alias("species"),
                    pl.n_unique("plot_id").alias("# plots"),
                    pl.n_unique("tree_id").alias("# trees"),
                    pl.len().alias("# observations"),
                ),
            ]
        ).sort("species")

    with pl.Config() as cfg:
        cfg.set_tbl_formatting("ASCII_MARKDOWN")
        cfg.set_tbl_hide_column_data_types(True)
        print(df)


def print_time_granularity(df: pl.DataFrame, keys: list[str]) -> None:
    entity_col = pl.col("tree_id") if "tree_id" in df.columns else pl.col("plot_id")
    if "date" in df.columns:
        time_col = pl.col("date")
        dt_mean = (
            df.sort(time_col, *keys)
            .with_columns(
                time_granularity=(
                    time_col.over(entity_col) - time_col.shift().over(entity_col)
                ).dt.total_days()
            )
            .select("time_granularity")
            .describe()
            .filter(pl.col("statistic") == "50%")
            .item(0, "time_granularity")
        )
    else:
        time_start_col = pl.col("date_start")
        time_end_col = pl.col("date_end")
        dt_mean = (
            df.with_columns(
                period_duration=(time_end_col - time_start_col).dt.total_days()
            )
            .select("period_duration")
            .describe()
            .filter(pl.col("statistic") == "50%")
            .item(0, "period_duration")
        )

    print(f"Time granularity for {', '.join(keys)}: {dt_mean:.2f} days")


if __name__ == "__main__":
    # Load raw data
    with pl.StringCache():
        df_growth_raw = pl.read_parquet("./data/raw/icpf-level2_growth.parquet")
        df_plots_raw = pl.read_parquet("./data/raw/icpf-level2_plot-info.parquet")
        df_crown_raw = pl.read_parquet(
            "./data/raw/icpf-level2_crown-conditions.parquet"
        )
        df_deposition_raw = pl.read_parquet(
            "./data/raw/icpf-level2_depositions.parquet"
        ).join(
            pl.read_parquet("./data/raw/icpf-level2_depositions_pld.parquet"),
            on="dem_pld_key",
            how="left",
        )
        df_soil_raw = pl.read_parquet("./data/raw/icpf-level2_soil-solutions.parquet")

    print("Growth:")
    print_time_granularity(df_growth_raw, keys=["tree_id"])
    print("Crown conditions:")
    print_time_granularity(df_crown_raw, keys=["tree_id"])
    print("Depositions:")
    print_time_granularity(
        df_deposition_raw.filter(pl.col("sampler_code") == 1),
        keys=["plot_id", "sampler_id"],
    )
    print("Soil solutions:")
    print_time_granularity(
        df_soil_raw.filter(pl.col("sample_vol").is_null() | pl.col("sample_vol").gt(0)),
        keys=["plot_id", "sampler_number"],
    )

    # --- Growth data ---
    print(f"Initial number of rows: {df_growth_raw.height}")

    df_growth = df_growth_raw.drop_nulls(subset="diameter")
    print(f" `- after dropping nulls: {df_growth.height}")

    df_growth = df_growth.filter(~pl.col("country").is_in(["Belgium", "Spain"]))
    print(f" `- after dropping Belgium and Spain: {df_growth.height}")

    df_growth = df_growth.filter(
        pl.col("diameter_quality_code").is_null()
        | ~pl.col("diameter_quality_code").gt(2)
    )
    df_growth = df_growth.filter(
        pl.col("diameter_method_code").is_null()
        | ~pl.col("diameter_method_code").is_in([7])
    )
    df_growth = df_growth.filter(
        pl.col("removal_code").is_null() | ~pl.col("removal_code").gt(10)
    )
    print(f" `- after dropping quality codes 3-9: {df_growth.height}")

    df_growth = df_growth.filter(pl.col("diameter").gt(0))
    print(f" `- after dropping negative diameters: {df_growth.height}")

    df_growth = (
        df_growth.sort(by=["country_code", "plot_code", "tree_number", "date"])
        .with_columns(
            period_start=pl.col("date")
            .shift(1)
            .over("country", "plot_code", "tree_number"),
            period_end=pl.col("date"),
            diameter_start=pl.col("diameter")
            .shift(1)
            .over("country", "plot_code", "tree_number"),
            diameter_end=pl.col("diameter"),
            diameter_method_code_start=pl.col("diameter_method_code")
            .shift(1)
            .over("country", "plot_code", "tree_number"),
            diameter_method_code_end=pl.col("diameter_method_code"),
        )
        .with_columns(
            period_duration=pl.col("period_end") - pl.col("period_start"),
            growth=pl.col("diameter_end") - pl.col("diameter_start"),
        )
        .with_columns(
            period_duration_d=pl.col("period_duration").dt.total_days(),
            period_duration_y=pl.col("period_duration").dt.total_days() / 365.25,
        )
        .with_columns(
            growth_rate=pl.col("growth") / pl.col("period_duration_y"),
            growth_rel=pl.col("growth") / pl.col("diameter_start"),
        )
        .with_columns(
            growth_rate_rel=pl.col("growth_rel") / pl.col("period_duration_y"),
        )
        .select(
            "survey_year",
            "tree_id",
            "plot_id",
            "country_code",
            "country",
            "tree_species_code",
            "specie",
            "plot_code",
            "tree_number",
            "period_start",
            "period_end",
            "diameter_start",
            "diameter_end",
            "period_duration_d",
            "period_duration_y",
            "growth",
            "growth_rate",
            "growth_rel",
            "growth_rate_rel",
            "diameter_method_code_start",
            "diameter_method_code_end",
            pl.col("removal_code").alias("removal_code_end"),
            pl.col("diameter_quality").alias("diameter_quality_end"),
            pl.col("diameter_method").alias("diameter_method_end"),
            pl.col("removal_info").alias("removal_info_end"),
        )
        .drop_nulls(subset=["period_duration_y"])
    )
    print(f" `- after computing growth rates: {df_growth.height}")

    df_growth = df_growth.filter(pl.col("period_duration_y").is_between(4.0, 6.0))
    print(
        f" `- after filtering growth periods between 4 and 6 years: {df_growth.height}"
    )

    df_growth = df_growth.filter(pl.col("growth_rate_rel").is_between(0, 0.1))
    print(
        f" `- after filtering relative growth rates between 0 and 0.1: {df_growth.height}"
    )

    print()
    print("Growth data summary:")
    summarize_data(df_growth)

    # --- Plot information ---
    PLOT_COLS = [
        "plot_latitude",
        "plot_longitude",
        "plot_slope",
        "plot_orientation",
        "plot_altitude",
    ]

    df_growth = df_growth.join(
        df_plots_raw.select("plot_id", *PLOT_COLS),
        on="plot_id",
        how="left",
    ).filter(pl.any_horizontal(cs.by_name(*PLOT_COLS).is_not_null()))

    print(f"Number of rows after joining with plot information: {df_growth.height}")
    print()
    print("Growth data summary:")
    summarize_data(df_growth)

    # --- Crown conditions ---
    print(f"Number of rows in crown condition data: {df_crown_raw.height}")

    df_crown = df_crown_raw.filter(
        pl.col("defoliation").is_not_null() & pl.col("defoliation").ge(0)
    ).with_columns(defoliation=pl.col("defoliation").cast(pl.Int32))
    print(f"Number of rows with valid defoliation: {df_crown.height}")

    df_crown = (
        df_crown.sort(by="date")
        .join_asof(
            df_growth.select(
                pl.col("period_end").alias("date"),
                "tree_id",
                "period_start",
                "period_end",
                "specie",
            ).sort(by="date"),
            by=["tree_id", "specie"],
            on="date",
            strategy="forward",
            suffix="_gp",
        )
        .filter(pl.col("date").is_between(pl.col("period_start"), pl.col("period_end")))
        .drop_nulls(subset="period_end")
    )
    print(f"Number of rows after merging crown condition data: {df_crown.height}")

    df_crown = df_crown.group_by("tree_id", "period_start", "period_end", "specie").agg(
        pl.len().alias("num_defoliation_obs"),
        pl.mean("defoliation").alias("defoliation_mean"),
        pl.min("defoliation").alias("defoliation_min"),
        pl.max("defoliation").alias("defoliation_max"),
        pl.median("defoliation").alias("defoliation_median"),
        pl.last("defoliation").alias("defoliation_last"),
        pl.min("social_class_code").alias("social_class_min"),
        pl.max("social_class_code").alias("social_class_max"),
        pl.col("social_class_code").mode().first().alias("social_class_mode"),
        pl.last("social_class_code").alias("social_class_last"),
        pl.col("social_class_code").eq(1).any().alias("was_dominant"),
        pl.col("social_class_code").eq(2).any().alias("was_codominant"),
        pl.col("social_class_code").eq(3).any().alias("was_subdominant"),
        pl.col("social_class_code").eq(4).any().alias("was_suppressed"),
        pl.col("social_class_code").eq(5).any().alias("was_dying"),
    )
    print(f"Number of rows after aggregating crown condition data: {df_crown.height}")

    df_crown = df_crown.filter(pl.col("defoliation_max").lt(100))
    print(f"Number of rows after dropping dead trees: {df_crown.height}")

    df_crown = df_crown.filter(pl.col("num_defoliation_obs").gt(1))
    print(
        f"Number of rows after dropping trees with less than two observations: {df_crown.height}"
    )

    df_growth_all = df_growth.join(
        df_crown,
        on=["tree_id", "period_start", "period_end", "specie"],
        how="left",
    ).drop_nulls(subset="num_defoliation_obs")

    print(f"Number of rows after merging crown condition data: {df_growth_all.height}")
    summarize_data(df_growth_all)

    # --- Depositions ---
    non_conc_cols = ["dep_alk", "dep_ph", "dep_cond"]

    print(f"Number of rows in deposition data {df_deposition_raw.height}")

    df_deposition = df_deposition_raw.filter(
        (pl.col("date_start").is_not_null() & pl.col("date_end").is_not_null())
        & (pl.col("sampler_code") == 1)
    )
    print(f" `- after dropping rows without census dates: {df_deposition.height}")

    df_deposition = df_deposition.filter(
        ~pl.col("vsampling_code").is_in([2, 3, 4, 7, 9])
    )
    print(f" `- after dropping abnormal sampling: {df_deposition.height}")

    df_deposition = df_deposition.filter(~pl.col("sampler_code").eq(8))
    print(f" `- after dropping sampler code 8: {df_deposition.height}")

    df_deposition = df_deposition.with_columns(
        pl.when(cs.starts_with("dep_").exclude(*non_conc_cols).ne(-1.0))
        .then(cs.starts_with("dep_").exclude(*non_conc_cols))
        .otherwise(None)
    ).with_columns(cs.starts_with("dep_").fill_nan(None))

    df_deposition = df_deposition.with_columns(
        dep_n_tot=pl.when(pl.col("dep_n_tot").is_null())
        .then(
            pl.col("dep_n_nh4") + pl.col("dep_n_no3") + pl.col("dep_n_org").fill_null(0)
        )
        .otherwise(pl.col("dep_n_tot"))
    )

    df_deposition = df_deposition.with_columns(
        (cs.starts_with("dep_").exclude(*non_conc_cols) * pl.col("quantity"))
    ).with_columns(cs.starts_with("dep_").exclude(*non_conc_cols) / 100)

    df_deposition = df_deposition.group_by("plot_id", "survey_year").agg(
        cs.starts_with("dep_").exclude(*non_conc_cols).sum(),
        cs.by_name(*non_conc_cols).mean(),
        pl.col("quantity").sum().alias("yearly_precip"),
        pl.len().alias("num_deposition_obs"),
    )

    summarize_data(df_deposition)

    df_deposition = (
        df_deposition.join(
            df_growth_all.select(
                "period_start", "period_end", "tree_id", "plot_id", "specie"
            ),
            on="plot_id",
            how="inner",
        )
        .with_columns(
            period_start_year=pl.col("period_start").dt.year(),
            period_end_year=pl.col("period_end").dt.year(),
        )
        .filter(
            pl.col("survey_year").is_between(
                pl.col("period_start_year"), pl.col("period_end_year")
            )
        )
        .group_by("tree_id", "period_start", "period_end")
        .agg(
            cs.starts_with("dep_").exclude(*non_conc_cols).sum(),
            cs.by_name(*non_conc_cols).mean(),
            pl.sum("num_deposition_obs").alias("num_deposition_obs"),
            pl.mean("yearly_precip").alias("yearly_precip"),
        )
        .with_columns(
            cs.starts_with("dep_").exclude(*non_conc_cols)
            / (pl.col("period_end").dt.year() - pl.col("period_start").dt.year()),
        )
    )

    print(
        f"Number of rows after joining deposition data with growth data: {df_deposition.height}"
    )

    df_growth_all = df_growth_all.join(
        df_deposition.select(
            "tree_id",
            "period_start",
            "period_end",
            "num_deposition_obs",
            "yearly_precip",
            "dep_ph",
            "dep_cond",
            "dep_k",
            "dep_ca",
            "dep_mg",
            "dep_na",
            "dep_n_nh4",
            "dep_cl",
            "dep_n_no3",
            "dep_s_so4",
            "dep_alk",
            "dep_n_tot",
            "dep_doc",
            "dep_al",
            "dep_mn",
            "dep_fe",
            "dep_p_po4",
            "dep_cu",
            "dep_zn",
            "dep_hg",
            "dep_pb",
            "dep_co",
            "dep_mo",
            "dep_ni",
            "dep_cd",
            "dep_s_tot",
            "dep_c_tot",
            "dep_n_org",
            "dep_p_tot",
            "dep_cr",
            "dep_n_no2",
            "dep_hco3",
            "dep_don",
        ),
        on=["tree_id", "period_start", "period_end"],
        how="left",
    ).with_columns(
        num_deposition_obs=pl.col("num_deposition_obs").fill_null(0),
        yearly_precip=pl.when(pl.col("yearly_precip") >= 250)
        .then(pl.col("yearly_precip"))
        .otherwise(pl.lit(None)),
    )

    print(f"Number of rows with at le: {df_growth_all.height}")
    summarize_data(df_growth_all)

    # --- Soil solutions ---
    print("Number of rows in raw soil solutions data:", df_soil_raw.height)

    df_soil = (
        df_soil_raw.filter(pl.col("sample_vol").is_null() | pl.col("sample_vol").gt(0))
        .with_columns(
            pl.when(cs.starts_with("ss_").is_between(0.0001, 10000))
            .then(cs.starts_with("ss_"))
            .otherwise(None)
        )
        .group_by("plot_id", "survey_year")
        .agg(
            pl.mean("sample_vol").alias("sample_vol"),
            cs.starts_with("ss_").mean().name.keep(),
            pl.len().alias("num_soil_obs"),
        )
        .select(
            "plot_id",
            "survey_year",
            "sample_vol",
            "num_soil_obs",
            "ss_ph",
            "ss_cond",
            "ss_k",
            "ss_ca",
            "ss_mg",
            "ss_n_no3",
            "ss_s_so4",
            "ss_alk",
            "ss_al",
            "ss_doc",
            "ss_na",
            "ss_n_nh4",
            "ss_cl",
            "ss_n_tot",
            "ss_fe",
            "ss_mn",
            "ss_al_labile",
            "ss_p",
            "ss_cr",
            "ss_ni",
            "ss_zn",
            "ss_cu",
            "ss_pb",
            "ss_cd",
            "ss_si",
        )
    )

    summarize_data(df_soil)

    df_soil_with_period = (
        df_soil.join(
            df_growth_all.select("plot_id", "tree_id", "period_start", "period_end"),
            on="plot_id",
            how="inner",
        )
        .with_columns(
            period_start_year=pl.col("period_start").dt.year(),
            period_end_year=pl.col("period_end").dt.year(),
        )
        .filter(
            pl.col("survey_year").is_between(
                pl.col("period_start_year"), pl.col("period_end_year")
            )
        )
        .group_by("tree_id", "period_start", "period_end")
        .agg(
            cs.starts_with("ss_").mean(),
            pl.sum("num_soil_obs").alias("num_soil_obs"),
        )
        .select(
            "tree_id",
            "period_start",
            "period_end",
            "num_soil_obs",
            cs.starts_with("ss_"),
        )
    )

    df_growth_all = df_growth_all.join(
        df_soil_with_period,
        on=["tree_id", "period_start", "period_end"],
        how="left",
    )

    print("Number of rows after merging soil solutions data:", df_growth_all.height)

    # --- Plot metadata (Etzold et al. 2023) ---
    df_plot_meta = (
        pl.read_csv("./data/raw/ICP-Forests-Plots_Meta.csv")
        .with_columns(plot_id=pl.col("plotid").cast(pl.Utf8).replace("NA", None))
        .drop_nulls(subset="plot_id")
        .with_columns(
            plot_id=pl.col("plot_id").str.slice(0, 2)
            + "."
            + pl.col("plot_id").str.slice(2),
            yr_first=pl.col("yr_first").replace("NA", None).cast(pl.Int32),
            yr_last=pl.col("yr_last").replace("NA", None).cast(pl.Int32),
            sdi=pl.col("sdi").replace("NA", None).cast(pl.Float32),
            age=pl.col("age").replace("NA", None).cast(pl.Float32),
            temp=pl.col("temp").replace("NA", None).cast(pl.Float32),
            precip=pl.col("precip").replace("NA", None).cast(pl.Float32),
        )
        .drop_nulls(subset=["yr_first", "yr_last"])
        .group_by("plot_id")
        .agg(
            pl.mean("sdi").alias("soph_avg_sdi"),
            pl.mean("age").alias("soph_avg_age"),
            pl.mean("temp").alias("soph_avg_temp"),
            pl.mean("precip").alias("soph_avg_precip"),
        )
    )

    summarize_data(df_plot_meta)

    df_growth_all = df_growth_all.join(df_plot_meta, on="plot_id", how="left")

    # --- Summary ---
    start_overall = df_growth_all.select(pl.col("period_start").min()).item()
    end_overall = df_growth_all.select(pl.col("period_end").max()).item()

    print(f"Start period: {start_overall}")
    print(f"End period: {end_overall}")
    print(f"Total duration (years): {(end_overall - start_overall).days / 365.25:.2f}")

    print()
    print("Statistics with crown conditions:")
    summarize_data(df_growth_all)

    print()
    print("Statistics with defoliation data:")
    summarize_data(
        df_growth_all.filter(pl.any_horizontal(cs.starts_with("dep_").is_not_null()))
    )

    print()
    print("Statistics with soil solution data:")
    summarize_data(
        df_growth_all.filter(pl.any_horizontal(cs.starts_with("ss_").is_not_null()))
    )

    # --- Table 1: Dataset statistics ---
    crown_plots = df_crown.join(
        df_growth_all.select("tree_id", "plot_id").unique(), on="tree_id", how="left"
    )["plot_id"].n_unique()
    dep_rows = df_growth_all.filter(
        pl.any_horizontal(cs.starts_with("dep_").is_not_null())
    )
    ss_rows = df_growth_all.filter(
        pl.any_horizontal(cs.starts_with("ss_").is_not_null())
    )
    meta_rows = df_growth_all.filter(pl.col("soph_avg_sdi").is_not_null())

    table1 = pl.DataFrame(
        {
            "Survey": [
                "Tree growth",
                "Crown condition",
                "Atmospheric deposition",
                "Soil solution chemistry",
                "Plot metadata",
            ],
            "Granularity": [
                "Tree-level",
                "Tree-level",
                "Plot-level",
                "Plot-level",
                "Plot-level",
            ],
            "Frequency": [
                "every 5 years",
                "annually",
                "every two weeks",
                "every two weeks",
                "static",
            ],
            "# plots": [
                df_growth["plot_id"].n_unique(),
                crown_plots,
                dep_rows["plot_id"].n_unique(),
                ss_rows["plot_id"].n_unique(),
                meta_rows["plot_id"].n_unique(),
            ],
            "# trees": [
                df_growth["tree_id"].n_unique(),
                df_crown["tree_id"].n_unique(),
                dep_rows["tree_id"].n_unique(),
                ss_rows["tree_id"].n_unique(),
                meta_rows["tree_id"].n_unique(),
            ],
            "Missing data allowed?": ["No", "No", "Yes", "Yes", "Yes"],
        }
    )

    print()
    print("Table 1: Dataset statistics")
    with pl.Config() as cfg:
        cfg.set_tbl_formatting("ASCII_MARKDOWN")
        cfg.set_tbl_hide_column_data_types(True)
        print(table1)

    # --- Species statistics ---
    DISPLAY_NAMES = {
        "spruce": "Norway spruce (Picea abies)",
        "pine": "Scots pine (Pinus sylvestris)",
        "beech": "Common beech (Fagus sylvatica)",
        "oak": "Sessile & Pedunculate oak (Quercus petraea and Quercus robur)",
    }

    species_stats = (
        df_growth_all.with_columns(
            species=pl.col("specie").cast(pl.Utf8).replace(SPECIES_MAPPING)
        )
        .filter(pl.col("species").is_in(SPECIES_MAPPING.values()))
        .group_by("species")
        .agg(
            pl.n_unique("tree_id").alias("trees"),
            pl.len().alias("growth_periods"),
        )
        .sort("growth_periods", descending=True)
    )

    print()
    print("Species statistics:")
    for row in species_stats.iter_rows(named=True):
        name = DISPLAY_NAMES[row["species"]]
        print(
            f"  {name}: {row['trees']:,} trees and {row['growth_periods']:,} growth periods"
        )

    # --- Write output ---
    df_growth_all.write_parquet(
        os.path.join(DATA_PATH, "tidy", "cpf-level2_cleaned.parquet")
    )

    features = pl.from_dicts(
        [
            {
                "feature": k,
                "Description": v["label"],
                "unit": v["unit"],
                "level": v["level"],
            }
            for k, v in FEATURES_METADATA.items()
        ]
    )

    with pl.Config() as cfg:
        cfg.set_tbl_formatting("ASCII_MARKDOWN")
        cfg.set_tbl_hide_column_data_types(True)
        cfg.set_tbl_rows(-1)
        print(features)
