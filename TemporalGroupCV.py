import numpy as np
import polars as pl
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import indexable
from sklearn.model_selection import GroupKFold, KFold
from data import prepare_data, load_data
from models import to_numpy
from typing import Optional, Tuple, Generator

class TemporalGroupCV(BaseCrossValidator):
    """
    Temporal Group Cross-Validator with period-based splitting.
    """
    
    def __init__(
        self, 
        df: pl.DataFrame, 
        cv: int = 5, 
        train_ratio: float = 0.8, 
        region_group: str = "plot_id", 
        time_start: str = "period_start", 
        time_end: str = "period_end", 
        group_by: Optional[str] = None, # "tree_id"
        random_state: int = 42
    ):
        """
        Parameters
        ----------
        df : pl.DataFrame
            DataFrame with temporal data
        cv : int, default=5
            Number of cross-validation folds
        train_ratio : float, default=0.8
            Ratio of periods to use for training
        region_group : str, default="plot_id"
            Column name for region/plot grouping
        time_start : str, default="period_start"
            Column name for start time
        time_end : str, default="period_end"
            Column name for end time
        group_by : str or None, default=None
            Column name for additional grouping in CV
        random_state : int, default=42
            Random state for reproducibility
        """
        self.df = df
        self.cv = cv
        self.train_ratio = train_ratio
        self.group_by = group_by
        self.random_state = random_state
        self.region_group = region_group
        self.time_start = time_start
        self.time_end = time_end
        
        self.train_indices_: np.ndarray
        self.test_indices_: np.ndarray
        self.train_indices_, self.test_indices_ = self._temporal_split(df)

    def _temporal_split(self, df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute train/test indices based on temporal order.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (train_indices, test_indices)
        """
        
        dfs = []
        unique_regions = df[self.region_group].unique().sort()
        for region_id in unique_regions:
            plot_df = df.filter(pl.col(self.region_group) == region_id)
            
            intervals = (plot_df
                .group_by([self.time_start, self.time_end])
                .agg(pl.len().alias("count"))
                .sort([self.time_start])
                .with_columns(
                    pl.int_range(1, pl.len() + 1).alias("period_idx")
                )
            ).drop("count")
            
            if len(intervals) == 0:
                continue
                
            plot_df = plot_df.join(intervals, on=[self.time_start, self.time_end])
            plot_df = plot_df.with_columns(
                pl.lit(len(intervals)).alias("tot_period_idx")
            )
            dfs.append(plot_df)
        
        if not dfs:
            raise ValueError("No data found after temporal splitting")
            
        df_processed = pl.concat(dfs)
        df_with_idx = df_processed.with_row_index("original_idx")
        
        # Get train/test indices
        train_mask = df_with_idx["period_idx"] <= (df_with_idx["tot_period_idx"] * self.train_ratio)
        test_mask = df_with_idx["period_idx"] > (df_with_idx["tot_period_idx"] * self.train_ratio)
        
        train_indices = df_with_idx.filter(train_mask)["original_idx"].to_numpy()
        test_indices = df_with_idx.filter(test_mask)["original_idx"].to_numpy()
        
        print(f"Train indices: {len(train_indices)}, Test indices: {len(test_indices)}")
        
        return train_indices, test_indices
        
    def get_n_splits(
        self, 
        X: Optional[np.ndarray] = None, 
        y: Optional[np.ndarray] = None, 
        groups: Optional[np.ndarray] = None
    ) -> int:
        """Returns the number of splitting iterations in the cross-validator"""
        return self.cv
    
    def temp_group_split(
        self, 
        X: np.ndarray, 
        y: Optional[np.ndarray] = None, 
        groups: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices to split data into training and test sets.
        """
        splitter = GroupKFold(n_splits=cv) if group_by else KFold(n_splits=cv)
        
        for fold, (train_idx, test_idx) in enumerate(splitter.split(to_numpy(X), y, groups=to_numpy(groups))):
            temp_train = np.intersect1d(train_idx, self.train_indices_)
            temp_test = np.intersect1d(test_idx, self.test_indices_)
            
            if len(temp_train) > 0 and len(temp_test) > 0:
                yield temp_train, temp_test
            else:
                print(f"Warning: Empty split in fold {fold} - train: {len(temp_train)}, test: {len(temp_test)}")
    
    def get_train_test_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the base temporal train/test indices"""
        return self.train_indices_, self.test_indices_


if __name__ == "__main__":
    species = "spruce"
    group_by = "tree_id"
    ablation = "all"
    cv = 5

    df = load_data(species)
    X, y, dist_params = prepare_data(df, ablation)
    
    if group_by is not None:
        groups = df.select(group_by).to_series().to_numpy()
    else:
        groups = None
    
    print("Temporal CV split:")
    splitter = TemporalGroupCV(
        df=df,
        train_ratio=0.8,
        cv=5,
        group_by=group_by,
        random_state=42
    )
    
    for fold, (train_idx, test_idx) in enumerate(splitter.temp_group_split(to_numpy(X), to_numpy(y), groups=groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        print(f"Fold {fold}: {len(train_idx)} {len(test_idx)}")
    
    print("\nGroup K Split")
    splitter = GroupKFold(n_splits=cv) if group_by else KFold(n_splits=cv)
    cv_splits = splitter.split(to_numpy(X), to_numpy(y), groups=groups)
    
    for fold, (train_idx, test_idx) in enumerate(cv_splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        print(f"Fold {fold}: {len(train_idx)} {len(test_idx)}")
    