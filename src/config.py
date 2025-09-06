DATA_DIR = "data"
RAW_DATA_DIR = f"{DATA_DIR}/raw"
PROCESSED_DATA_DIR = f"{DATA_DIR}/processed"
MODELS_DIR = "models"
SUBMISSIONS_DIR = "submissions"

TRAIN_FILE = f"{RAW_DATA_DIR}/train.csv"
TEST_FILE = f"{RAW_DATA_DIR}/test.csv"
METADATA_FILE = f"{RAW_DATA_DIR}/metadata.csv"
SUBMISSION_FILE = f"{SUBMISSIONS_DIR}/submission_xgboost.csv"

TARGET_COL = "meter_reading"
MODEL_PARAMS = {
    'objective': 'reg:squarederror',
    'n_estimators': 7000,
    'learning_rate': 0.007,
    'max_depth': 8,
    'subsample': 0.95,
    'colsample_bytree': 0.9,
    'random_state': 42,
    'n_jobs': -1,
    'early_stopping_rounds': 150
}

LAG_FEATURES = [1, 48, 168]
ROLLING_WINDOW_FEATURES = [6, 12, 24, 168]