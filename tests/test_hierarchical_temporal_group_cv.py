"""Tests for HierarchicalTimeGroupCV cross-validator."""

import sys
import logging
from itertools import combinations
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import polars as pl
import pytest
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, str(project_root))

# Stub external project modules before importing the class under test
_models_mock = MagicMock()
_models_mock.to_numpy = np.asarray
sys.modules.setdefault("data", MagicMock())
sys.modules.setdefault("models", _models_mock)

from HierarchicalTemporaGroupCV import HierarchicalTimeGroupCV  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cv():
    return HierarchicalTimeGroupCV(n_splits_tree=3, log_level=logging.WARNING)


def _make_polars_df(n_plots=3, n_periods_per_plot=3, n_trees_per_cell=4):
    """Build a minimal polars DataFrame matching the expected schema."""
    rows = []
    tree_id = 0
    for plot in range(1, n_plots + 1):
        for period in range(n_periods_per_plot):
            start = 2000 + period * 5
            end = start + 4
            for _ in range(n_trees_per_cell):
                rows.append(
                    {
                        "plot_id": plot,
                        "tree_id": tree_id,
                        "period_start": start,
                        "period_end": end,
                        "value": float(tree_id),
                    }
                )
                tree_id += 1
    return pl.DataFrame(rows)


def _make_split_inputs(n_plots=3, n_periods=3, n_trees_per_cell=4, n_splits=3):
    """Return (X, y, tree_groups, period_group, plot_group) as numpy arrays."""
    rows = []
    for plot in range(n_plots):
        for period in range(1, n_periods + 1):
            for tree in range(n_trees_per_cell):
                tree_id = (
                    plot * n_periods * n_trees_per_cell
                    + (period - 1) * n_trees_per_cell
                    + tree
                )
                rows.append((plot, period, tree_id))

    plot_group = np.array([r[0] for r in rows])
    period_group = np.array([r[1] for r in rows])
    tree_groups = np.array([r[2] for r in rows])

    n = len(rows)
    X = np.random.default_rng(42).random((n, 5))
    y = np.random.default_rng(0).random(n)
    return X, y, tree_groups, period_group, plot_group


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_n_splits(self):
        cv = HierarchicalTimeGroupCV()
        assert cv.n_splits_tree == 5

    def test_custom_n_splits(self):
        cv = HierarchicalTimeGroupCV(n_splits_tree=7)
        assert cv.n_splits_tree == 7

    def test_logger_created(self, cv):
        assert cv.logger is not None


# ---------------------------------------------------------------------------
# get_n_splits
# ---------------------------------------------------------------------------


class TestGetNSplits:
    def test_returns_n_splits_tree(self, cv):
        assert cv.get_n_splits() == 3

    def test_ignores_all_arguments(self, cv):
        assert (
            cv.get_n_splits(X=np.zeros((10, 2)), y=np.zeros(10), groups=np.zeros(10))
            == 3
        )


# ---------------------------------------------------------------------------
# add_period_indices
# ---------------------------------------------------------------------------


class TestAddPeriodIndices:
    def test_column_added(self, cv):
        df = _make_polars_df()
        result = cv.add_period_indices(df)
        assert "period_idx" in result.columns

    def test_period_idx_starts_at_one(self, cv):
        df = _make_polars_df()
        result = cv.add_period_indices(df)
        for plot_id in result["plot_id"].unique():
            min_idx = result.filter(pl.col("plot_id") == plot_id)["period_idx"].min()
            assert min_idx == 1

    def test_period_idx_is_contiguous_per_plot(self, cv):
        df = _make_polars_df(n_plots=2, n_periods_per_plot=4)
        result = cv.add_period_indices(df)
        for plot_id in result["plot_id"].unique():
            indices = sorted(
                result.filter(pl.col("plot_id") == plot_id)["period_idx"]
                .unique()
                .to_list()
            )
            assert indices == list(range(1, len(indices) + 1))

    def test_row_count_preserved(self, cv):
        df = _make_polars_df()
        result = cv.add_period_indices(df)
        assert len(result) == len(df)

    def test_single_period_plot(self, cv):
        df = pl.DataFrame(
            {
                "plot_id": [1, 1, 1],
                "tree_id": [10, 11, 12],
                "period_start": [2000, 2000, 2000],
                "period_end": [2004, 2004, 2004],
            }
        )
        result = cv.add_period_indices(df)
        assert result["period_idx"].to_list() == [1, 1, 1]


# ---------------------------------------------------------------------------
# generate_period_splits
# ---------------------------------------------------------------------------


class TestGeneratePeriodSplits:
    def test_all_periods_covered(self, cv):
        periods = [1, 2, 3, 4]
        n_train = 2
        for train, test in cv.generate_period_splits(periods, n_train):
            assert train | test == set(periods)
            assert train & test == set()

    def test_train_size_correct(self, cv):
        periods = [1, 2, 3, 4]
        n_train = 2
        for train, _ in cv.generate_period_splits(periods, n_train):
            assert len(train) == n_train

    def test_number_of_splits(self, cv):
        periods = [1, 2, 3, 4]
        n_train = 2
        splits = list(cv.generate_period_splits(periods, n_train))
        expected = len(list(combinations(periods, n_train)))
        assert len(splits) == expected

    def test_no_empty_test_sets(self, cv):
        periods = [1, 2, 3]
        for _, test in cv.generate_period_splits(periods, 2):
            assert len(test) > 0

    def test_all_periods_as_train_yields_nothing(self, cv):
        periods = [1, 2, 3]
        splits = list(cv.generate_period_splits(periods, 3))
        assert splits == []


# ---------------------------------------------------------------------------
# temporal_split
# ---------------------------------------------------------------------------


class TestTemporalSplit:
    def test_yields_n_splits_folds(self, cv):
        X, y, tree_groups, period_group, plot_group = _make_split_inputs(n_splits=3)
        folds = list(cv.temporal_split(X, y, tree_groups, period_group, plot_group))
        assert len(folds) == cv.n_splits_tree

    def test_train_test_are_disjoint(self, cv):
        X, y, tree_groups, period_group, plot_group = _make_split_inputs()
        for train_idx, test_idx in cv.temporal_split(
            X, y, tree_groups, period_group, plot_group
        ):
            assert set(train_idx).isdisjoint(set(test_idx))

    def test_indices_within_bounds(self, cv):
        X, y, tree_groups, period_group, plot_group = _make_split_inputs()
        n = len(y)
        for train_idx, test_idx in cv.temporal_split(
            X, y, tree_groups, period_group, plot_group
        ):
            assert all(0 <= i < n for i in train_idx)
            assert all(0 <= i < n for i in test_idx)

    def test_accepts_dataframe_input(self, cv):
        X, y, tree_groups, period_group, plot_group = _make_split_inputs()
        X_df = pd.DataFrame(X)
        folds = list(cv.temporal_split(X_df, y, tree_groups, period_group, plot_group))
        assert len(folds) == cv.n_splits_tree

    def test_single_period_plots_are_skipped(self, cv):
        """Plots with only one period should not contribute any indices."""
        rng = np.random.default_rng(1)
        n = 20
        X = rng.random((n, 3))
        y = rng.random(n)
        # All samples belong to a single plot with a single period
        plot_group = np.zeros(n, dtype=int)
        period_group = np.ones(n, dtype=int)
        tree_groups = np.arange(n)

        cv3 = HierarchicalTimeGroupCV(n_splits_tree=3, log_level=logging.WARNING)
        for train_idx, test_idx in cv3.temporal_split(
            X, y, tree_groups, period_group, plot_group
        ):
            assert len(train_idx) == 0
            assert len(test_idx) == 0

    def test_non_overlapping_tree_groups_across_train_test(self, cv):
        """Tree-level GroupKFold must separate trees between train and test."""
        X, y, tree_groups, period_group, plot_group = _make_split_inputs()
        for train_idx, test_idx in cv.temporal_split(
            X, y, tree_groups, period_group, plot_group
        ):
            train_trees = set(tree_groups[train_idx])
            test_trees = set(tree_groups[test_idx])
            assert train_trees.isdisjoint(test_trees), (
                "Tree groups should not overlap between train and test"
            )

    def test_no_coplot_coperiod_trees_split_across_train_and_test(self, cv):
        """No two trees sharing the same (plot_id, period_id) may appear on
        opposite sides of a fold — one in train, the other in test."""
        X, y, tree_groups, period_group, plot_group = _make_split_inputs()
        plot_group = np.asarray(plot_group)
        period_group = np.asarray(period_group)
        tree_groups = np.asarray(tree_groups)

        for fold_idx, (train_idx, test_idx) in enumerate(
            cv.temporal_split(X, y, tree_groups, period_group, plot_group)
        ):
            train_idx = np.asarray(train_idx)
            test_idx = np.asarray(test_idx)

            # Map each tree_id to the side of the split it landed on.
            train_trees = set(tree_groups[train_idx])
            test_trees = set(tree_groups[test_idx])

            # Group tree_ids by (plot_id, period_id).
            from collections import defaultdict

            cell_to_trees: dict = defaultdict(set)
            for i in range(len(tree_groups)):
                cell_to_trees[(plot_group[i], period_group[i])].add(tree_groups[i])

            for (plot_id, period_id), trees_in_cell in cell_to_trees.items():
                cell_in_train = trees_in_cell & train_trees
                cell_in_test = trees_in_cell & test_trees
                assert not (cell_in_train and cell_in_test), (
                    f"Fold {fold_idx}: trees {cell_in_train} (train) and "
                    f"{cell_in_test} (test) share plot_id={plot_id}, "
                    f"period_id={period_id}"
                )
