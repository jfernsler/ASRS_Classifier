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

# data without labels
ASRS_RAW_DATA = os.path.join(DATA_DIR, 'raw', 'asrs.csv')
ASRS_DATA_CLEAN = os.path.join(DATA_DIR, 'raw', 'asrs_clean.pkl.zip')

# data with labels
ASRS_LABELED_DATA = os.path.join(PICKLE_DATA, 'asrs_cluster_labels.pkl.zip')
# pre-split data with labels
ASRS_CLEAN_TRAIN = os.path.join(PICKLE_DATA, 'asrs_data_clean_2000_train.pkl.zip')
ASRS_CLEAN_TEST = os.path.join(PICKLE_DATA, 'asrs_data_clean_2000_test.pkl.zip')

# data with labels and embeddings
ASRS_HF_TEST = os.path.join(PICKLE_DATA, 'asrs_HF_test_2000.pkl.zip')
ASRS_HF_TRAIN = os.path.join(PICKLE_DATA, 'asrs_HF_train_2000.pkl.zip')