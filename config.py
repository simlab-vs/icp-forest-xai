from typing import Literal

# Define key constants
Species = Literal["spruce", "pine", "beech", "oak"]
Levels = Literal["tree", "plot"]

# Paths
DATA_PATH = "./data"

# Features
FEATURES_DESCRIPTION = {
    # "diameter_end": {
    #     "description": "Diameter at the end of the period",
    #     "level": "tree",
    #     "unit": "cm",
    # },
    "defoliation_max": {
        "description": "Maximum defoliation of the growth period",
        "descriptive_name": "Max defoliation",
        "level": "tree",
        "unit": "%",
    },
    "defoliation_min": {
        "description": "Minimum defoliation of the growth period",
        "descriptive_name": "Min defoliation",
        "level": "tree",
        "unit": "%",
    },
    "defoliation_mean": {
        "description": "Mean defoliation of the growth period",
        "descriptive_name":"Mean defoliation",
        "level": "tree",
        "unit": "%",
    },
    "defoliation_median": {
        "description": "Median defoliation of the growth period",
        "descriptive_name":"Median defoliation",
        "level": "tree",
        "unit": "%",
    },
    "social_class_min": {
        "description": "Minimum social class of the growth period",
        "descriptive_name": "Min social class",
        "level": "tree",
        "unit": None,
    },
    # "country",
    "plot_latitude": {
        "description": "Latitude of the plot",
        "descriptive_name": "Latitude",
        "level": "plot",
        "unit": "°",
    },
    "plot_longitude": {
        "description": "Longitude of the plot",
        "descriptive_name": "Longitude",
        "level": "plot",
        "unit": "°",
    },
    "plot_slope": {
        "description": "Slope of the plot",
        "descriptive_name": "Slope",
        "level": "plot",
        "unit": "%"
    },
    "plot_orientation": {
        "description": "Orientation of the plot",
        "descriptive_name": "Plot Orientation",
        "level": "plot",
        "unit": None,
    },
    "plot_altitude": {
        "description": "Altitude of the plot",
        "descriptive_name": "Altitude",
        "level": "plot",
        "unit": "m",
    },
    "dep_ph": {
        "description": "Deposition pH",
        "descriptive_name": "Deposition pH",
        "level": "plot",
        "unit": None
    },
    "dep_cond": {
        "description": "Deposition conductivity",
        "descriptive_name": "Deposition conductivity",
        "level": "plot",
        "unit": "µS/cm",
    },
    "dep_k": {
        "description": "Deposition potassium (K)",
        "descriptive_name": "Deposition potassium",
        "level": "plot",
        "unit": "mg/l",
    },
    "dep_ca": {
        "description": "Deposition calcium (Ca)",
        "descriptive_name": "Deposition calcium",
        "level": "plot",
        "unit": "mg/l",
    },
    "dep_mg": {
        "description": "Deposition magnesium (Mg)",
        "descriptive_name": "Deposition magnesium",
        "level": "plot",
        "unit": "mg/l",
    },
    "dep_na": {
        "description": "Deposition sodium (Na)",
        "descriptive_name": "Deposition sodium",
        "level": "plot",
        "unit": "mg/l",
    },
    "dep_n_nh4": {
        "description": "Deposition ammonium (NH4)",
        "descriptive_name": "Deposition ammonium",
        "level": "plot",
        "unit": "mg/l",
    },
    "dep_cl": {
        "description": "Deposition chloride (Cl)",
        "descriptive_name": "Deposition chloride",
        "level": "plot",
        "unit": "mg/l",
    },
    "dep_n_no3": {
        "description": "Deposition nitrate (NO3)",
        "descriptive_name": "Deposition nitrate",
        "level": "plot",
        "unit": "mg/l",
    },
    "dep_s_so4": {
        "description": "Deposition sulfate (SO4)",
        "descriptive_name":"Deposition sulfate",
        "level": "plot",
        "unit": "mg/l",
    },
    "dep_alk": {
        "description": "Deposition alkalinity",
        "descriptive_name": "Deposition alkalinity",
        "level": "plot",
        "unit": "µEq/l",
    },
    "dep_n_tot": {
        "description": "Deposition total nitrogen (N)",
        "descriptive_name": "Deposition nitrogen",
        "level": "plot",
        "unit": "mg/l",
    },
    "dep_doc": {
        "description": "Deposition dissolved organic carbon (DOC)",
        "descriptive_name": "Deposition dissolved organic carbon",
        "level": "plot",
        "unit": "mg/l",
    },
    "dep_al": {
        "description": "Deposition aluminium (Al)",
        "descriptive_name": "Deposition aluminium",
        "level": "plot",
        "unit": "mg/l",
    },
    "dep_mn": {
        "description": "Deposition manganese (Mn)",
        "descriptive_name": "Deposition manganese",
        "level": "plot",
        "unit": "mg/l",
    },
    "dep_fe": {
        "description": "Deposition iron (Fe)",
        "descriptive_name": "Deposition iron",
        "level": "plot",
        "unit": "mg/l"
    },
    "dep_p_po4": {
        "description": "Deposition phosphate (PO4)",
        "descriptive_name": "Deposition phosphate",
        "level": "plot",
        "unit": "mg/l",
    },
    "dep_cu": {
        "description": "Deposition copper (Cu)",
        "descriptive_name": "Deposition copper",
        "level": "plot",
        "unit": "µg/l",
    },
    "dep_zn": {
        "description": "Deposition zinc (Zn)",
        "descriptive_name": "Deposition zinc",
        "level": "plot",
        "unit": "µg/l"
    },
    "dep_hg": {
        "description": "Deposition mercury (Hg)",
        "descriptive_name": "Deposition mercury",
        "level": "plot",
        "unit": "µg/l",
    },
    "dep_pb": {
        "description": "Deposition lead (Pb)",
        "descriptive_name": "Deposition lead",
        "level": "plot",
        "unit": "µg/l"
    },
    "dep_co": {
        "description": "Deposition cobalt (Co)",
        "descriptive_name": "Deposition cobalt",
        "level": "plot",
        "unit": "µg/l",
    },
    # "dep_mo": {"description": "Deposition molybdenum (Mo)", "level": "plot", "unit": "µg/l"},
    "dep_ni": {
        "description": "Deposition nickel (Ni)",
        "descriptive_name": "Deposition nickel",
        "level": "plot",
        "unit": "µg/l",
    },
    "dep_cd": {
        "description": "Deposition cadmium (Cd)",
        "descriptive_name": "Deposition cadmium",
        "level": "plot",
        "unit": "µg/l",
    },
    "dep_s_tot": {
        "description": "Deposition total sulfur (S)",
        "descriptive_name": "Deposition sulfur",
        "level": "plot",
        "unit": "mg/l",
    },
    "dep_c_tot": {
        "description": "Deposition total carbon (C)",
        "descriptive_name": "Deposition carbon",
        "level": "plot",
        "unit": "mg/l",
    },
    # "dep_n_org": {"description": "Deposition organic nitrogen (N)", "level": "plot", "unit": "mg/l"},
    "dep_p_tot": {
        "description": "Deposition total phosphorus (P)",
        "descriptive_name": "Deposition phosphorus",
        "level": "plot",
        "unit": "mg/l",
    },
    "dep_cr": {
        "description": "Deposition chromium (Cr)",
        "descriptive_name": "Deposition chromium",
        "level": "plot",
        "unit": "µg/l",
    },
    "dep_n_no2": {
        "description": "Deposition nitrite (NO2)",
        "descriptive_name": "Deposition nitrate",
        "level": "plot",
        "unit": "mg/l",
    },
    # "dep_hco3": {"description": "Deposition bicarbonate (HCO3)", "level": "plot", "unit": "mg/l"},
    # "dep_don": {"description": "Deposition dissolved organic nitrogen (DON)", "level": "plot", "unit": "mg/l"},
    "ss_ph": {
        "description": "Soil solution pH",
        "descriptive_name": "Soil solution pH",
        "level": "plot",
        "unit": None
    },
    "ss_cond": {
        "description": "Soil solution conductivity",
        "descriptive_name": "Soil solution conductivity",
        "level": "plot",
        "unit": "µS/cm",
    },
    "ss_k": {
        "description": "Soil solution potassium (K)",
        "descriptive_name": "Soil solution conductivity",
        "level": "plot",
        "unit": "mg/l",
    },
    "ss_ca": {
        "description": "Soil solution calcium (Ca)",
        "descriptive_name": "Soil solution calcium",
        "level": "plot",
        "unit": "mg/l",
    },
    "ss_mg": {
        "description": "Soil solution magnesium (Mg)",
        "descriptive_name": "Soil solution magnesium",
        "level": "plot",
        "unit": "mg/l",
    },
    "ss_n_no3": {
        "description": "Soil solution nitrate (NO3)",
        "descriptive_name": "Soil solution nitrate",
        "level": "plot",
        "unit": "mg/l",
    },
    "ss_s_so4": {
        "description": "Soil solution sulfate (SO4)",
        "descriptive_name": "Soil solution sulfate",
        "level": "plot",
        "unit": "mg/l",
    },
    "ss_alk": {
        "description": "Soil solution alkalinity",
        "descriptive_name": "Soil solution alkalinity",
        "level": "plot",
        "unit": "µmolc/l",
    },
    "ss_al": {
        "description": "Soil solution aluminium (Al)",
        "descriptive_name": "Soil solution aluminium",
        "level": "plot",
        "unit": "mg/l",
    },
    "ss_doc": {
        "description": "Soil solution dissolved organic carbon (DOC)",
        "descriptive_name":"Soil solution dissolved organic carbon",
        "level": "plot",
        "unit": "mg/l",
    },
    "ss_na": {
        "description": "Soil solution sodium (Na)",
        "descriptive_name": "Soil solution sodium",
        "level": "plot",
        "unit": "mg/l",
    },
    "ss_n_nh4": {
        "description": "Soil solution ammonium (NH4)",
        "descriptive_name": "Soil solution ammonium",
        "level": "plot",
        "unit": "mg/l",
    },
    "ss_cl": {
        "description": "Soil solution chloride (Cl)",
        "descriptive_name": "Soil solution chloride",
        "level": "plot",
        "unit": "mg/l",
    },
    "ss_n_tot": {
        "description": "Soil solution total nitrogen (N)",
        "descriptive_name": "Soil solution nitrogen",
        "level": "plot",
        "unit": "mg/l",
    },
    "ss_fe": {
        "description": "Soil solution iron (Fe)",
        "descriptive_name": "Soil solution iron",
        "level": "plot",
        "unit": "mg/l",
    },
    "ss_mn": {
        "description": "Soil solution manganese (Mn)",
        "descriptive_name": "Soil solution manganese",
        "level": "plot",
        "unit": "mg/l",
    },
    "ss_al_labile": {
        "description": "Soil solution labile aluminium (Al)",
        "descriptive_name": "Soil solution labile aluminium",
        "level": "plot",
        "unit": "mg/l",
    },
    "ss_p": {
        "description": "Soil solution phosphorus (P)",
        "descriptive_name": "Soil solution phosphorus",
        "level": "plot",
        "unit": "mg/l",
    },
    "ss_cr": {
        "description": "Soil solution chromium (Cr)",
        "descriptive_name": "Soil solution chromium",
        "level": "plot",
        "unit": "µg/l",
    },
    "ss_ni": {
        "description": "Soil solution nickel (Ni)",
        "descriptive_name": "Soil solution nickel",
        "level": "plot",
        "unit": "µg/l",
    },
    "ss_zn": {
        "description": "Soil solution zinc (Zn)",
        "descriptive_name": "Soil solution zinc",
        "level": "plot",
        "unit": "µg/l",
    },
    "ss_cu": {
        "description": "Soil solution copper (Cu)",
        "descriptive_name": "Soil solution copper",
        "level": "plot",
        "unit": "µg/l",
    },
    "ss_pb": {
        "description": "Soil solution lead (Pb)",
        "descriptive_name": "Soil solution lead",
        "level": "plot",
        "unit": "µg/l",
    },
    "ss_cd": {
        "description": "Soil solution cadmium (Cd)",
        "descriptive_name": "Soil solution cadmium",
        "level": "plot",
        "unit": "µg/l",
    },
    "ss_si": {
        "description": "Soil solution silicon (Si)",
        "descriptive_name": "Soil solution silicon",
        "level": "plot",
        "unit": "mg/l",
    },
    "soph_avg_sdi": {
        "description": "Average species diversity index",
        "descriptive_name": "Average species diversity index",
        "level": "plot",
        "unit": None,
    },
    "soph_avg_age": {
        "description": "Average age of the trees",
        "descriptive_name": "Average age of the trees",
        "level": "plot",
        "unit": "years",
    },
    "soph_avg_temp": {
        "description": "Average temperature",
        "descriptive_name": "Average temperature",
        "level": "plot",
        "unit": "°C",
    },
    "soph_avg_precip": {
        "description": "Average precipitation",
        "descriptive_name": "Average precipitation",
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
