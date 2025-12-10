from typing import Literal

# Define key constants
Species = Literal["spruce", "pine", "beech", "oak"]
Levels = Literal["tree", "plot"]

# Paths
DATA_PATH = "./data"

# Features
FEATURES_METADATA = {
    # "diameter_end": {
    #     "description": "Diameter at the end of the period",
    #     "label": "Tree diameter",
    #     "level": "tree",
    #     "unit": "cm",
    # },
    "defoliation_max": {
        "description": "Maximum defoliation of the growth period",
        "label": "Max defoliation",
        "level": "tree",
        "unit": "%",
    },
    "defoliation_min": {
        "description": "Minimum defoliation of the growth period",
        "label": "Min defoliation",
        "level": "tree",
        "unit": "%",
    },
    "defoliation_mean": {
        "description": "Mean defoliation of the growth period",
        "label": "Mean defoliation",
        "level": "tree",
        "unit": "%",
    },
    "defoliation_median": {
        "description": "Median defoliation of the growth period",
        "label": "Median defoliation",
        "level": "tree",
        "unit": "%",
    },
    "social_class_min": {
        "description": "Minimum social class of the growth period",
        "label": "Min. social class",
        "level": "tree",
        "unit": None,
    },
    # "plot_latitude": {
    #     "description": "Latitude of the plot",
    #     "label": "Latitude",
    #     "level": "plot",
    #     "unit": "°",
    # },
    # "plot_longitude": {
    #     "description": "Longitude of the plot",
    #     "label": "Longitude",
    #     "level": "plot",
    #     "unit": "°",
    # },
    "plot_slope": {
        "description": "Slope of the plot",
        "label": "Plot slope",
        "level": "plot",
        "unit": "%",
    },
    "plot_orientation": {
        "description": "Orientation of the plot",
        "label": "Plot orientation",
        "level": "plot",
        "unit": None,
    },
    "plot_altitude": {
        "description": "Altitude of the plot",
        "label": "Plot altitude",
        "level": "plot",
        "unit": "m",
    },
    "yearly_precip": {
        "description": "Total precipitation",
        "label": "Cumul. precipitation",
        "level": "plot",
        "unit": "mm/yr",
    },
    "dep_ph": {
        "description": "Deposition pH",
        "label": "Dep. pH",
        "level": "plot",
        "unit": None,
    },
    "dep_cond": {
        "description": "Deposition conductivity",
        "label": "Dep. conductivity",
        "level": "plot",
        "unit": "µS/cm",
    },
    "dep_k": {
        "description": "Deposition potassium (K)",
        "label": "Dep. potassium (K)",
        "level": "plot",
        "unit": "mg/l",
    },
    "dep_ca": {
        "description": "Deposition calcium (Ca)",
        "label": "Dep. calcium (Ca)",
        "level": "plot",
        "unit": "mg/l",
    },
    "dep_mg": {
        "description": "Deposition magnesium (Mg)",
        "label": "Dep. magnesium (Mg)",
        "level": "plot",
        "unit": "mg/l",
    },
    "dep_na": {
        "description": "Deposition sodium (Na)",
        "label": "Dep. sodium (Na)",
        "level": "plot",
        "unit": "mg/l",
    },
    "dep_n_tot": {
        "description": "Deposition total nitrogen (N)",
        "label": "Dep. total nitrogen (N)",
        "level": "plot",
        "unit": "mg/l",
    },
    # "dep_n_nh4": {
    #     "description": "Deposition ammonium (NH4)",
    #     "label": "Dep. ammonium (NH4)",
    #     "level": "plot",
    #     "unit": "mg/l",
    # },
    # "dep_n_no3": {
    #     "description": "Deposition nitrate (NO3)",
    #     "label": "Dep. nitrate (NO3)",
    #     "level": "plot",
    #     "unit": "mg/l",
    # },
    # "dep_n_no2": {
    #     "description": "Deposition nitrite (NO2)",
    #     "label": "Dep. nitrite (NO2)",
    #     "level": "plot",
    #     "unit": "mg/l",
    # },
    # "dep_n_org": {
    #     "description": "Deposition organic nitrogen (N)",
    #     "label": "Dep. organic nitrogen (N)",
    #     "level": "plot",
    #     "unit": "mg/l",
    # },
    "dep_cl": {
        "description": "Deposition chloride (Cl)",
        "label": "Dep. chloride (Cl)",
        "level": "plot",
        "unit": "mg/l",
    },
    "dep_s_so4": {
        "description": "Deposition sulfate (SO4)",
        "label": "Dep. sulfate (SO4)",
        "level": "plot",
        "unit": "mg/l",
    },
    "dep_alk": {
        "description": "Deposition alkalinity",
        "label": "Dep. alkalinity",
        "level": "plot",
        "unit": "µEq/l",
    },
    "dep_doc": {
        "description": "Deposition dissolved organic carbon (DOC)",
        "label": "Dep. DOC",
        "level": "plot",
        "unit": "mg/l",
    },
    "dep_al": {
        "description": "Deposition aluminium (Al)",
        "label": "Dep. aluminium (Al)",
        "level": "plot",
        "unit": "mg/l",
    },
    "dep_mn": {
        "description": "Deposition manganese (Mn)",
        "label": "Dep. manganese (Mn)",
        "level": "plot",
        "unit": "mg/l",
    },
    "dep_fe": {
        "description": "Deposition iron (Fe)",
        "label": "Dep. iron (Fe)",
        "level": "plot",
        "unit": "mg/l",
    },
    "dep_cu": {
        "description": "Deposition copper (Cu)",
        "label": "Dep. copper (Cu)",
        "level": "plot",
        "unit": "µg/l",
    },
    "dep_zn": {
        "description": "Deposition zinc (Zn)",
        "label": "Dep. zinc (Zn)",
        "level": "plot",
        "unit": "µg/l",
    },
    "dep_hg": {
        "description": "Deposition mercury (Hg)",
        "label": "Dep. mercury (Hg)",
        "level": "plot",
        "unit": "µg/l",
    },
    "dep_pb": {
        "description": "Deposition lead (Pb)",
        "label": "Dep. lead (Pb)",
        "level": "plot",
        "unit": "µg/l",
    },
    "dep_co": {
        "description": "Deposition cobalt (Co)",
        "label": "Dep. cobalt (Co)",
        "level": "plot",
        "unit": "µg/l",
    },
    "dep_ni": {
        "description": "Deposition nickel (Ni)",
        "label": "Dep. nickel (Ni)",
        "level": "plot",
        "unit": "µg/l",
    },
    "dep_cd": {
        "description": "Deposition cadmium (Cd)",
        "label": "Dep. cadmium (Cd)",
        "level": "plot",
        "unit": "µg/l",
    },
    "dep_s_tot": {
        "description": "Deposition total sulfur (S)",
        "label": "Dep. total sulfur (S)",
        "level": "plot",
        "unit": "mg/l",
    },
    "dep_c_tot": {
        "description": "Deposition total carbon (C)",
        "label": "Dep. total carbon (C)",
        "level": "plot",
        "unit": "mg/l",
    },
    "dep_p_tot": {
        "description": "Deposition total phosphorus (P)",
        "label": "Dep. total phosphorus (P)",
        "level": "plot",
        "unit": "mg/l",
    },
    "dep_cr": {
        "description": "Deposition chromium (Cr)",
        "label": "Dep. chromium (Cr)",
        "level": "plot",
        "unit": "µg/l",
    },
    "ss_ph": {
        "description": "Soil solution pH",
        "label": "S.s. pH",
        "level": "plot",
        "unit": None,
    },
    "ss_cond": {
        "description": "Soil solution conductivity",
        "label": "S.s. conductivity",
        "level": "plot",
        "unit": "µS/cm",
    },
    "ss_k": {
        "description": "Soil solution potassium (K)",
        "label": "S.s. potassium (K)",
        "level": "plot",
        "unit": "mg/l",
    },
    "ss_ca": {
        "description": "Soil solution calcium (Ca)",
        "label": "S.s. calcium (Ca)",
        "level": "plot",
        "unit": "mg/l",
    },
    "ss_mg": {
        "description": "Soil solution magnesium (Mg)",
        "label": "S.s. magnesium (Mg)",
        "level": "plot",
        "unit": "mg/l",
    },
    "ss_n_no3": {
        "description": "Soil solution nitrate (NO3)",
        "label": "S.s. nitrate (NO3)",
        "level": "plot",
        "unit": "mg/l",
    },
    "ss_s_so4": {
        "description": "Soil solution sulphate (SO4)",
        "label": "S.s. sulphate (SO4)",
        "level": "plot",
        "unit": "mg/l",
    },
    "ss_alk": {
        "description": "Soil solution alkalinity",
        "label": "S.s. alkalinity",
        "level": "plot",
        "unit": "µmolc/l",
    },
    "ss_al": {
        "description": "Soil solution aluminium (Al)",
        "label": "S.s. aluminium (Al)",
        "level": "plot",
        "unit": "mg/l",
    },
    "ss_doc": {
        "description": "Soil solution dissolved organic carbon (DOC)",
        "label": "S.s. DOC",
        "level": "plot",
        "unit": "mg/l",
    },
    "ss_na": {
        "description": "Soil solution sodium (Na)",
        "label": "S.s. sodium (Na)",
        "level": "plot",
        "unit": "mg/l",
    },
    "ss_n_nh4": {
        "description": "Soil solution ammonium (NH4)",
        "label": "S.s. ammonium (NH4)",
        "level": "plot",
        "unit": "mg/l",
    },
    "ss_cl": {
        "description": "Soil solution chloride (Cl)",
        "label": "S.s. chloride (Cl)",
        "level": "plot",
        "unit": "mg/l",
    },
    "ss_n_tot": {
        "description": "Soil solution total nitrogen (N)",
        "label": "S.s. total nitrogen (N)",
        "level": "plot",
        "unit": "mg/l",
    },
    "ss_fe": {
        "description": "Soil solution iron (Fe)",
        "label": "S.s. iron (Fe)",
        "level": "plot",
        "unit": "mg/l",
    },
    "ss_mn": {
        "description": "Soil solution manganese (Mn)",
        "label": "S.s. manganese (Mn)",
        "level": "plot",
        "unit": "mg/l",
    },
    "ss_al_labile": {
        "description": "Soil solution labile aluminium (Al)",
        "label": "S.s. labile aluminium (Al)",
        "level": "plot",
        "unit": "mg/l",
    },
    "ss_p": {
        "description": "Soil solution phosphorus (P)",
        "label": "S.s. phosphorus (P)",
        "level": "plot",
        "unit": "mg/l",
    },
    "ss_cr": {
        "description": "Soil solution chromium (Cr)",
        "label": "S.s. chromium (Cr)",
        "level": "plot",
        "unit": "µg/l",
    },
    "ss_ni": {
        "description": "Soil solution nickel (Ni)",
        "label": "S.s. nickel (Ni)",
        "level": "plot",
        "unit": "µg/l",
    },
    "ss_zn": {
        "description": "Soil solution zinc (Zn)",
        "label": "S.s. zinc (Zn)",
        "level": "plot",
        "unit": "µg/l",
    },
    "ss_cu": {
        "description": "Soil solution copper (Cu)",
        "label": "S.s. copper (Cu)",
        "level": "plot",
        "unit": "µg/l",
    },
    "ss_pb": {
        "description": "Soil solution lead (Pb)",
        "label": "S.s. lead (Pb)",
        "level": "plot",
        "unit": "µg/l",
    },
    "ss_cd": {
        "description": "Soil solution cadmium (Cd)",
        "label": "S.s. cadmium (Cd)",
        "level": "plot",
        "unit": "µg/l",
    },
    "ss_si": {
        "description": "Soil solution silicon (Si)",
        "label": "S.s. silicon (Si)",
        "level": "plot",
        "unit": "mg/l",
    },
    "soph_avg_sdi": {
        "description": "Stand density index",
        "label": "SDI",
        "level": "plot",
        "unit": None,
    },
    "soph_avg_age": {
        "description": "Average age of the trees",
        "label": "Plot age",
        "level": "plot",
        "unit": "years",
    },
    "soph_avg_temp": {
        "description": "Average temperature",
        "label": "Mean temperature",
        "level": "plot",
        "unit": "°C",
    },
    "soph_avg_precip": {
        "description": "Average precipitation",
        "label": "Mean precipitation",
        "level": "plot",
        "unit": "mm",
    },
}

# Configure the features and target variable
TARGET = "growth_rate_rel"

# List ablations
Ablation = Literal[
    "all",
    "tree-level-only",
    "plot-level-only",
    "no-defoliation",
    "max-defoliation",
    "min-defoliation",
    "median-defoliation",
]


# Subet of columns that are categorical
CATEGORICAL_COLUMNS = ["country", "plot_orientation"]

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
