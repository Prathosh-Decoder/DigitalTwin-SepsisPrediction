"""Shared paths, constants, and hyperparameters for the sepsis prediction pipeline."""
from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATA_DIR_A = PROJECT_DIR / "training_setA" / "training_setA"
DATA_DIR_B = PROJECT_DIR / "training_setB" / "training_setB"

CACHE_DIR = BASE_DIR / "data_cache"
RAW_PARQUET = CACHE_DIR / "combined_raw.parquet"
FEATURES_PARQUET = CACHE_DIR / "features.parquet"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
MODEL_BUNDLE_PATH = ARTIFACTS_DIR / "model_bundle.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
SHAP_VALUES_PATH = ARTIFACTS_DIR / "shap_values.npy"
SHAP_SAMPLE_FEATURES_PATH = ARTIFACTS_DIR / "shap_sample_features.parquet"
SHAP_EXPLAINER_PATH = ARTIFACTS_DIR / "shap_explainer.joblib"
TRAIN_IDS_PATH = ARTIFACTS_DIR / "train_patient_ids.json"
VAL_IDS_PATH = ARTIFACTS_DIR / "val_patient_ids.json"
TEST_IDS_PATH = ARTIFACTS_DIR / "test_patient_ids.json"

for d in (CACHE_DIR, ARTIFACTS_DIR, PLOTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Reproducibility ---
RANDOM_SEED = 42

# --- Raw schema ---
VITAL_COLS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LAB_COLS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]
MEASURED_COLS = VITAL_COLS + LAB_COLS  # 8 + 26 = 34
STATIC_COLS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
LABEL_COL = "SepsisLabel"

# Curated subset used for rolling stats / deltas (7 core vitals + 5 sepsis-relevant labs)
ROLLING_VARS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp",
                 "WBC", "Lactate", "Creatinine", "Platelets", "Bilirubin_total"]
ROLLING_WINDOWS = [6, 24]
ROLLING_STATS = ["mean", "min", "max", "std"]

# Hours-since-measured sentinel for labs never yet drawn
HOURS_SINCE_SENTINEL = 336.0  # 14 days

# --- Partial SOFA cutoffs (ascending thresholds -> points 0-4) ---
SOFA_PLATELETS_CUTOFFS = [150, 100, 50, 20]  # higher points as platelets drop below each cutoff
SOFA_BILIRUBIN_CUTOFFS = [1.2, 2.0, 6.0, 12.0]  # higher points as bilirubin rises above each cutoff
SOFA_CREATININE_CUTOFFS = [1.2, 2.0, 3.5, 5.0]  # higher points as creatinine rises above each cutoff
SOFA_MAP_THRESHOLD = 70  # MAP < 70 -> 1 point (capped; no vasopressor dose data available)
SOFA_DELTA_WINDOW_HOURS = 24
SOFA_WORSENING_DELTA = 2

# --- Official PhysioNet/CinC 2019 utility function constants ---
DT_EARLY = -12
DT_OPTIMAL = -6
DT_LATE = 3
MAX_U_TP = 1.0
MIN_U_FN = -2.0
U_FP = -0.05
U_TN = 0.0

# --- Split ---
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1

# --- LightGBM hyperparameters ---
# Base values from the published winning ("Can I get your signature?") config; n_estimators
# and learning_rate were then adjusted via a light grid search (06_hyperparameter_search.py)
# over n_estimators/num_leaves/learning_rate on the existing train/val split. The chosen
# combo (300/49/0.05) improved BOTH validation utility (0.4285 -> 0.4354) and AUROC
# (0.8420 -> 0.8443) over the original published values, and kept num_leaves unchanged
# to minimize deviation from the winning team's config. See
# artifacts/hyperparam_search_results.csv for the full 27-combination search.
LGBM_PARAMS = dict(
    boosting_type="gbdt",
    num_leaves=49,
    max_depth=6,
    learning_rate=0.05,
    n_estimators=300,
    min_child_samples=122,
    min_child_weight=1,
    min_split_gain=0.0,
    reg_alpha=100,
    reg_lambda=0,
    subsample=0.3465,
    subsample_freq=1,  # must be >=1 for subsample to take effect in LightGBM
    colsample_bytree=0.5494,
    subsample_for_bin=200000,
    random_state=RANDOM_SEED,
    n_jobs=8,
    verbosity=-1,
)
EARLY_STOPPING_ROUNDS = 20

# --- Threshold grid search ---
THRESHOLD_GRID_MIN = -0.5
THRESHOLD_GRID_MAX = 0.5
THRESHOLD_GRID_STEP = 0.01

# --- SHAP sampling ---
SHAP_MAX_POSITIVES = 2000
SHAP_MAX_TOTAL = 10000
