import pandas as pd
from src import config

def load_data():
    print("Loading data...")
    try:
        train_df = pd.read_csv(config.TRAIN_FILE, parse_dates=['timestamp'])
        test_df = pd.read_csv(config.TEST_FILE, parse_dates=['timestamp'])
        metadata_df = pd.read_csv(config.METADATA_FILE)
        sample_submission_df = pd.read_csv(f"{config.RAW_DATA_DIR}/sample_submission.csv")
        
        print(f"Train data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")
        print(f"Metadata shape: {metadata_df.shape}")
        print(f"Sample submission shape: {sample_submission_df.shape}")
        
        return train_df, test_df, metadata_df, sample_submission_df
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure data files are in 'data/raw' directory.")
        raise