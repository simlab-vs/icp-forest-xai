"""Train and explain models for all species for a given configuration.

Usage
-----
    uv run train.py --model-type gbdt --ablation all --group-col tree_id --temporal-cv

Results are written to ./cache/:
  - results-{ablation}-{model_type}-{group_col}.pkl          (joblib, all ExperimentResults)
  - feature_importances-{ablation}-{model_type}-{group_col}.parquet
"""

import argparse
import os

import joblib
import numpy as np
import polars as pl
import polars.selectors as cs
from sklearn.metrics import r2_score, root_mean_squared_error

from config import Ablation
from models import ALL_SPECIES, ExperimentResults, ModelType, Species, train_and_explain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--group-col",
        default="tree_id",
        choices=["tree_id", "plot_id", "none"],
        help="Grouping column for K-fold CV (default: tree_id)",
    )
    parser.add_argument(
        "--model-type",
        default="gbdt",
        choices=["gbdt", "elasticnet"],
        help="Model type (default: gbdt)",
    )
    parser.add_argument(
        "--ablation",
        default="all",
        choices=[
            "all",
            "tree-level-only",
            "plot-level-only",
            "no-defoliation",
            "max-defoliation",
            "min-defoliation",
            "median-defoliation",
        ],
        help="Ablation study variant (default: all)",
    )
    parser.add_argument(
        "--temporal-cv",
        action="store_true",
        help="Use hierarchical temporal group CV",
    )
    return parser.parse_args()


def _print_summary(all_results: dict[Species, ExperimentResults]) -> None:
    rows = []
    for species, results in all_results.items():
        for fold in range(results.num_folds):
            X_train, y_true_train, y_pred_train = results.get_data(fold, "train")
            X_test, y_true_test, y_pred_test = results.get_data(fold, "test")
            rows.append(
                {
                    "species": species,
                    "r2_train": r2_score(y_true_train, y_pred_train),
                    "r2_test": r2_score(y_true_test, y_pred_test),
                    "rmse_train": root_mean_squared_error(y_true_train, y_pred_train),
                    "rmse_test": root_mean_squared_error(y_true_test, y_pred_test),
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                }
            )

    df = pl.DataFrame(rows)
    summary = df.group_by("species").agg(
        [
            pl.col("r2_test").mean().alias("R2_mean"),
            pl.col("r2_test").std().alias("R2_std"),
            pl.col("rmse_test").mean().alias("RMSE_mean"),
            pl.col("rmse_test").std().alias("RMSE_std"),
            pl.col("n_test").first(),
        ]
    )

    total_test = float(df["n_test"].sum())
    weighted_r2 = float((df["r2_test"] * df["n_test"]).sum()) / total_test
    weighted_rmse = float((df["rmse_test"] * df["n_test"]).sum()) / total_test

    print(f"\n{'Species':<12} | {'R2 (test)':<22} | {'RMSE (test)':<22} | n_test")
    print("-" * 72)
    for row in summary.sort("species").iter_rows(named=True):
        r2_str = f"{row['R2_mean']:.3f} ± {row['R2_std']:.3f}"
        rmse_str = f"{row['RMSE_mean']:.3f} ± {row['RMSE_std']:.3f}"
        print(f"{row['species']:<12} | {r2_str:<22} | {rmse_str:<22} | {row['n_test']}")
    print("-" * 72)
    print(f"{'Weighted':<12} | {weighted_r2:.3f}{'':<19} | {weighted_rmse:.3f}")


def _save_hyperparams(
    all_results: dict[Species, ExperimentResults],
    ablation: str,
    model_type: str,
    group_col: str | None,
) -> None:
    rows = []
    for species, results in all_results.items():
        for fold, estimator in enumerate(results.estimators):
            get_hp = getattr(estimator, "get_hyperparams", None)
            if get_hp is None:
                continue
            rows.append({"species": species, "fold": fold, **get_hp()})

    if not rows:
        return

    path = f"./cache/hyperparams-{ablation}-{model_type}-{group_col}.parquet"
    pl.DataFrame(rows).write_parquet(path)
    print(f"Hyperparameters saved to {path}")


def main() -> None:
    args = parse_args()

    group_col: str | None = None if args.group_col == "none" else args.group_col
    model_type: ModelType = args.model_type
    ablation: Ablation = args.ablation
    use_temporal_cv: bool = args.temporal_cv

    os.makedirs("./cache", exist_ok=True)

    print(
        f"Config: model={model_type}, ablation={ablation}, group={group_col}, temporal_cv={use_temporal_cv}\n"
    )

    all_results: dict[Species, ExperimentResults] = {}
    for species in ALL_SPECIES:
        all_results[species] = train_and_explain(
            species,
            model_type=model_type,
            group_by=group_col,
            ablation=ablation,
            use_temporal_cv=use_temporal_cv,
        )

    results_path = f"./cache/results-{ablation}-{model_type}-{group_col}.pkl"
    joblib.dump(all_results, results_path)
    print(f"\nResults saved to {results_path}")

    num_folds = next(iter(all_results.values())).num_folds
    feature_importances = pl.from_dicts(
        [
            {
                "species": species,
                "fold": fold,
                **dict(
                    zip(
                        results.features,
                        np.abs(results.shap_values[fold].values).mean(axis=0),
                    )
                ),
                "n": len(results.shap_values[fold].values),
            }
            for species, results in all_results.items()
            for fold in range(num_folds)
        ]
    ).unpivot(
        on=cs.exclude("species", "fold", "n"),
        index=["species", "fold", "n"],
        variable_name="feature",
        value_name="shap",
    )

    fi_path = f"./cache/feature_importances-{ablation}-{model_type}-{group_col}.parquet"
    feature_importances.write_parquet(fi_path)
    print(f"Feature importances saved to {fi_path}")

    _save_hyperparams(all_results, ablation, model_type, group_col)

    _print_summary(all_results)


if __name__ == "__main__":
    main()
