"""Hierarchical Temporal Group Cross-Validation for Forest Growth Data"""

import random
import numpy as np
import math
import pandas as pd
import polars as pl
from itertools import combinations
from sklearn.model_selection import BaseCrossValidator, GroupKFold
from data import prepare_data, load_data

import logging

# Set to logging.WARNING or logging.WARNING to suppress all INFO messages
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class HierarchicalTimeGroupCV(BaseCrossValidator):
    """
    Cross-validator for hierarchical time series data with grouping by trees and plots.

    This validator creates splits that respect temporal ordering within plots and
    tree groupings across folds.

    Parameters
    ----------
    n_splits_tree : int, default=5
        Number of splits for the tree-level GroupKFold cross-validation.
    log_level : int, default=logging.INFO
        Logging level for the cross-validator. Set to logging.WARNING to suppress messages.
    """

    def __init__(self, n_splits_tree=5, log_level=logging.INFO, random_state=None):
        """
        Initialize the HierarchicalTimeGroupCV cross-validator.

        Parameters
        ----------
        n_splits_tree : int, default=5
            Number of splits for the tree-level GroupKFold cross-validation.
        log_level : int, default=logging.INFO
            Logging level for the cross-validator. Set to logging.WARNING to suppress messages.
        """
        self.n_splits_tree = n_splits_tree
        self.logger = logging.getLogger(f"{__name__}.HierarchicalTimeGroupCV")
        self.logger.setLevel(log_level)
        self.random_state = random_state
        self._rng = random.Random(random_state)

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Return the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), optional
            Training data.
        y : array-like of shape (n_samples,), optional
            Target values.
        groups : array-like of shape (n_samples,), optional
            Group labels for the samples.

        Returns
        -------
        int
            Number of splits.
        """

        return self.n_splits_tree

    def add_period_indices(self, df):
        """
        Add period indices to the dataframe based on temporal ordering within each plot.

        Parameters
        ----------
        df : pl.DataFrame
            Input polars DataFrame containing plot_id, period_start, and period_end columns.

        Returns
        -------
        pl.DataFrame
            DataFrame with added 'period_idx' column.
        """
        self.logger.info("Adding period indices to dataframe")

        dfs = []
        for plot_id in df["plot_id"].unique():
            plot_df = df.filter(pl.col("plot_id") == plot_id)
            intervals = (
                plot_df.group_by(["period_start", "period_end"])
                .agg(pl.len().alias("tree_count"))
                .sort(["period_start"])
                .with_columns(pl.int_range(1, pl.len() + 1).alias("period_idx"))
            )
            plot_df = plot_df.join(intervals, on=["period_start", "period_end"])
            dfs.append(plot_df)

        result_df = pl.concat(dfs)
        self.logger.info("Created dataframe with period_idx column")
        return result_df

    def prepare_cv_data(
        self,
        species,
        ablation="all",
        tree_group="tree_id",
        period_group="period_idx",
        plot_group="plot_id",
    ):
        """
        Load and prepare data for cross-validation.

        Parameters
        ----------
        species : str
            Species name (e.g., "spruce", "pine").
        ablation : str, default="all"
            Ablation study parameter for prepare_data function.
        tree_group : str, default="tree_id"
            Column name for tree group identifiers.
        period_group : str, default="period_idx"
            Column name for period indices.
        plot_group : str, default="plot_id"
            Column name for plot identifiers.

        Returns
        -------
        tuple
            (X, y, tree_groups, period_groups, plot_groups, dist_params)
        """
        self.logger.info(f"Loading data for species: {species}")

        # Load raw data
        df = load_data(species)

        # Add period indices
        df = self.add_period_indices(df)

        # Prepare features and targets
        X, y, _dist_params = prepare_data(df, ablation)

        # Extract group labels
        tree_groups = df.select(tree_group).to_series().to_numpy()
        period_groups = df.select(period_group).to_series().to_numpy()
        plot_groups = df.select(plot_group).to_series().to_numpy()

        self.logger.info(
            f"Data prepared: X shape={X.shape if hasattr(X, 'shape') else 'unknown'}, "
            f"n_samples={len(y)}"
        )

        return X, y, tree_groups, period_groups, plot_groups

    def generate_period_splits(self, periods, n_train_periods):
        """
        Generate all possible train/test combinations of time periods.

        Parameters
        ----------
        periods : List[int]
            List of available time periods.
        n_train_periods : int
            Number of periods to include in the training set.

        Yields
        ------
        Tuple[Set[int], Set[int]]
            A tuple containing (train_set, test_set) where both are sets of period indices.
        """
        for train_combo in combinations(periods, n_train_periods):
            train_set = set(train_combo)
            test_set = set(periods) - train_set

            if len(test_set) == 0:
                continue

            yield train_set, test_set

    def temporal_split(self, X, y, tree_groups, period_group, plot_group):
        """
        Generate train/test indices for temporal cross-validation.

        This method creates splits that respect both tree groupings (via GroupKFold)
        and temporal ordering within plots.

        Parameters
        ----------
        X : Union[np.ndarray, pd.DataFrame]
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Target values of shape (n_samples,).
        tree_groups : np.ndarray
            Tree group labels for each sample.
        period_group : np.ndarray
            Period indices for each sample.
        plot_group : np.ndarray
            Plot IDs for each sample.

        Returns
        ------
        Tuple[List[int], List[int]]
            A tuple containing (train_indices, test_indices) for each fold.

        """

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        gkf = GroupKFold(n_splits=self.n_splits_tree)

        unique_plots = np.unique(np.asarray(plot_group))

        # For each tree fold
        for fold, (train_tree_idx, test_tree_idx) in enumerate(
            gkf.split(X, y, groups=tree_groups)
        ):
            train_idxs, test_idxs = [], []

            # Split growth periods using plot_ids
            for plot_id in unique_plots:
                plot_mask = np.asarray(plot_group) == plot_id
                periods = list(np.unique(np.asarray(period_group)[plot_mask]))

                # Leaves out plots with just one growth periods.
                if len(periods) == 1:
                    continue

                plot_splits = list(
                    self.generate_period_splits(periods, math.ceil(len(periods) / 2))
                )

                train_periods, test_periods = self._rng.choice(plot_splits)

                train_mask = (
                    plot_mask
                    & np.isin(np.arange(len(X)), train_tree_idx)
                    & np.isin(np.asarray(period_group), list(train_periods))
                )

                test_mask = (
                    plot_mask
                    & np.isin(np.arange(len(X)), test_tree_idx)
                    & np.isin(np.asarray(period_group), list(test_periods))
                )

                train_idx = np.where(train_mask)[0]
                test_idx = np.where(test_mask)[0]

                train_idxs.extend(train_idx)
                test_idxs.extend(test_idx)

            yield train_idxs, test_idxs

    def run_cross_validation(
        self,
        species,
        ablation="all",
        tree_group="tree_id",
        period_group="period_idx",
        plot_group="plot_id",
    ):
        """
        Run complete cross-validation pipeline.

        This method loads data, prepares it, and performs temporal cross-validation.

        Parameters
        ----------
        species : str
            Species name ("spruce", "pine", "beech", "oak").
        ablation : str, default="all"
            Ablation study parameter for prepare_data function.
        tree_group : str, default="tree_id"
            Column name for tree group identifiers.
        period_group : str, default="period_idx"
            Column name for period indices.
        plot_group : str, default="plot_id"
            Column name for plot identifiers.

        Yields
        ------
        Tuple[List[int], List[int]]
            A tuple containing (train_indices, test_indices) for each fold.
        """
        # Prepare data
        X, y, tree_groups, period_groups, plot_groups = self.prepare_cv_data(
            species=species,
            ablation=ablation,
            tree_group=tree_group,
            period_group=period_group,
            plot_group=plot_group,
        )

        # Perform temporal splits
        for fold, (train_idx, test_idx) in enumerate(
            self.temporal_split(
                X,
                y,
                tree_groups=tree_groups,
                period_group=period_groups,
                plot_group=plot_groups,
            )
        ):
            self.logger.info(
                f"Fold {fold} - Train periods: {len(train_idx)}, Test periods: {len(test_idx)}"
            )
            yield train_idx, test_idx

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples,), default=None
            Target values.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples

        Returns
        -------
        Tuple[List[int], List[int]]
        """
        tree_groups, period_groups, plot_groups = None, None, None

        if isinstance(groups, tuple) and len(groups) == 3:
            tree_groups, period_groups, plot_groups = groups

        for train_idx, test_idx in self.temporal_split(
            X, y, tree_groups, period_groups, plot_groups
        ):
            yield np.array(train_idx), np.array(test_idx)


if __name__ == "__main__":
    # Example usage
    cv = HierarchicalTimeGroupCV(n_splits_tree=5, log_level=logging.ERROR)

    for fold, (train_idx, test_idx) in enumerate(
        cv.run_cross_validation(species="spruce")
    ):
        print(f"Fold {fold}: train={len(train_idx)}, test={len(test_idx)}")
