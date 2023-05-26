from pathlib import Path
import os

# overview paths
MOD_DIR = os.path.dirname(Path(__file__).absolute())
PROJ_PATH = os.path.join(MOD_DIR, '..', '..')

# subdir paths
DATA_DIR = os.path.join(PROJ_PATH, 'data')
DATA_OUT_DIR = os.path.join(PROJ_PATH, 'data_out')
PICKLE_DATA = os.path.join(DATA_DIR, 'pickle_data')

# output paths
CHART_DIR = os.path.join(PROJ_PATH, 'charts')
MODEL_DIR = os.path.join(PROJ_PATH, 'models')

# version paths
MODEL_PATH_TUNED = os.path.join(MODEL_DIR, 'ASRS_distilbert-base-uncased')
MODEL_BEST_CHECKPOINT = 37563

ASRS_DATA = os.path.join(DATA_DIR, 'asrs.csv')
ASRS_DATA_CLEAN = os.path.join(DATA_DIR, 'asrs_clean.csv')
ASRS_DATA_CLEAN_PKL = os.path.join(DATA_DIR, 'asrs_clean.pkl')