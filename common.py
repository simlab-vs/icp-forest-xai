from typing import Literal

# Define key constants
Species = Literal["spruce", "pine", "beech", "oak"]

# Paths
DATA_PATH = "./data"

# Scoring metrics
SCORING_METRICS = ["r2", "neg_mean_absolute_error"]

# Features
FEATURES = [
    "diameter_end",
    "defoliation_max",
    "defoliation_mean",
    "social_class_min",
    # "country",
    "plot_latitude",
    "plot_slope",
    "plot_orientation",
    "plot_altitude",
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
    # "dep_mo",
    "dep_ni",
    "dep_cd",
    "dep_s_tot",
    "dep_c_tot",
    "dep_n_org",
    "dep_p_tot",
    "dep_cr",
    "dep_n_no2",
    # "dep_hco3",
    # "dep_don",
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
    "soph_avg_sdi",
    "soph_avg_age",
    "soph_avg_temp",
    "soph_avg_precip",
]

TARGET = "growth_rate_rel"

# Subet of columns that are categorical
CATEGORICAL_COLUMNS = [
    "country",
    "plot_orientation"
]

# These parameters have been tuned using Optuna
LIGHTGBM_PARAMS = {
    "spruce": {
        "num_leaves": 34,
        "max_depth": 13,
        "min_data_in_leaf": 24,
        "lambda_l1": 1.81e-05,
        "lambda_l2": 0.0004,
        "min_gain_split": 0.82,
        "feature_fraction": 0.48,
        "bagging_fraction": 0.91,
        "bagging_freq": 3,
    },
    "pine": {
        "num_leaves": 111,
        "max_depth": 8,
        "min_data_in_leaf": 61,
        "lambda_l1": 0.044,
        "lambda_l2": 0.00019,
        "min_gain_split": 0.51,
        "feature_fraction": 0.64,
        "bagging_fraction": 0.80,
        "bagging_freq": 4,
    },
    "beech": {
        "num_leaves": 85,
        "max_depth": 6,
        "min_data_in_leaf": 23,
        "lambda_l1": 0.0013,
        "lambda_l2": 1.92e-07,
        "min_gain_split": 0.998,
        "feature_fraction": 0.40,
        "bagging_fraction": 0.83,
        "bagging_freq": 5,
    },
    "oak": {
        "num_leaves": 181,
        "max_depth": 9,
        "min_data_in_leaf": 35,
        "lambda_l1": 2.48e-06,
        "lambda_l2": 0.0057,
        "min_gain_split": 0.54,
        "feature_fraction": 0.55,
        "bagging_fraction": 0.67,
        "bagging_freq": 2,
    },
}
