import os

# ROOT (project root, not src)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# DATA
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

DATA_PATH = os.path.join(RAW_DATA_DIR, "housing.csv")

# OUTPUTS
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

# PARAMETERS
RANDOM_STATE = 42
TEST_SIZE = 0.2

# SDR PARAMETERS
N_NEIGHBORS_SDR = 50
N_BOOTSTRAP_SDR = 30
LAMBDA_SDR = 1.0
SHAP_BACKGROUND_SIZE = 100