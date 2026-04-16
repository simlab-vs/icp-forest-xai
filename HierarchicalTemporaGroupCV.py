import random
import numpy as np
import pandas as pd
import polars as pl
from itertools import combinations
from sklearn.model_selection import BaseCrossValidator, GroupKFold
from sklearn.model_selection import GroupKFold
from data import prepare_data, load_data
from models import to_numpy

class HierarchicalTimeGroupCV(BaseCrossValidator):
    def __init__(
        self,
        n_splits_tree=5,
    ):
        self.n_splits_tree = n_splits_tree
     
    def get_n_splits(self, X=None, y=None, groups=None):
        # Approximate
        return self.n_splits_tree
    
    def generate_period_splits(self, periods, n_train_periods):
        
        for train_combo in combinations(periods, n_train_periods):
            train_set = set(train_combo)
            test_set = set(periods) - train_set
            
            if len(test_set) == 0:
                continue
                
            yield train_set, test_set
        
    def temporal_split(self, X, y, tree_groups, period_group, plot_group):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
          
        gkf = GroupKFold(n_splits=self.n_splits_tree)
        
        unique_plots = np.unique(np.asarray(plot_group))
        
        # For each tree fold 
        for fold, (train_tree_idx, test_tree_idx) in enumerate(gkf.split(X, y, groups=to_numpy(tree_groups))):
            
            print(f"Tree fold {fold}")
            print(f"Length of train: {len(train_tree_idx)}, Length of test: {len(test_tree_idx)}")
            train_idxs, test_idxs = [], []
            
            # Split growth periods using plot_ids
            for plot_id in unique_plots:
                plot_mask = np.isin(np.asarray(plot_group), [plot_id])
                plot_periods = np.unique(np.asarray(period_group)[plot_mask])
                periods = sorted(list(plot_periods))
            
                plot_splits = list(self.generate_period_splits(periods, round(len(periods) * 0.6)))
                
                # Leaves out plots with just one growth periods. 
                if len(plot_splits) == 0:
                    continue
            
                train_periods, test_periods = random.choice(plot_splits)

                train_mask = (
                    plot_mask &
                    np.isin(np.arange(len(X)), train_tree_idx) &
                    np.isin(np.asarray(period_group), list(train_periods))
                )

                test_mask = (
                    plot_mask &
                    np.isin(np.arange(len(X)), test_tree_idx) &  
                    np.isin(np.asarray(period_group), list(test_periods))
                )
                
                train_idx = np.where(train_mask)[0]
                test_idx = np.where(test_mask)[0]
                
                train_idxs.extend(train_idx)
                test_idxs.extend(test_idx)
        
            yield train_idxs, test_idxs
                
if __name__ == "__main__":
    
    species = "spruce"
    tree_group = "tree_id"
    period_group = "period_idx"
    plot_group = "plot_id"
    ablation = "all"
    cv = 5

    df = load_data(species)

    dfs = []
    for plot_id in df["plot_id"].unique():
        plot_df = df.filter(pl.col("plot_id") == plot_id)
        
        intervals = (plot_df
            .group_by(["period_start", "period_end"])
            .agg(pl.len().alias("tree_count"))
            .sort(["period_start"])
            .with_columns(
                pl.int_range(1, pl.len() + 1).alias("period_idx")
            )
        )

        plot_df = plot_df.join(intervals, on=["period_start", "period_end"])
        
        plot_df = plot_df.with_columns(
            pl.lit(len(intervals)).alias("tot_period_idx")
        )
        dfs.append(plot_df)
    
    df = pl.concat(dfs)
    df_with_idx = df.with_row_index("original_idx")
    
    X, y, dist_params = prepare_data(df, ablation)
    
    tree_groups = df.select(tree_group).to_series()
    period_groups = df.select(period_group).to_series()
    plot_groups = df.select(plot_group).to_series()
   
    print("Created dataframe with period_idx column")
    
    cv = HierarchicalTimeGroupCV(
        n_splits_tree=5
    )

    for fold, (train_idx, test_idx) in enumerate(cv.temporal_split(
                        to_numpy(X), 
                        y, 
                        tree_groups = to_numpy(tree_groups), 
                        period_group = to_numpy(period_groups),
                        plot_group = to_numpy(plot_groups)
                    )):
        
        print("Train periods:", "Length:", len(train_idx))
        print("Test periods:", "Length:", len(test_idx))
        print("\n")