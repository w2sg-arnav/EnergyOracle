import pandas as pd
import os
import sys
import argparse
from src.data_loader import load_data
from src.model_trainer import train_model
from src.predict import make_predictions
from src import config

def main(sample_size=None):
    print("Starting the EnergyOracle pipeline...")
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.SUBMISSIONS_DIR, exist_ok=True)

    print("Loading data...")
    try:
        train_df, test_df, metadata_df, sample_submission_df = load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("Training model...")
    try:
        model, features = train_model(train_df, metadata_df, sample_size=sample_size)
    except Exception as e:
        print(f"Error during model training: {e}")
        return

    print("Generating predictions...")
    try:
        submission_df = make_predictions(test_df, metadata_df, model, features)
        print(f"Submission shape: {submission_df.shape}")
        if submission_df.shape[0] != 84600:
            print(f"Warning: Submission has {submission_df.shape[0]} rows, expected 84600")
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    print("EnergyOracle pipeline finished successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EnergyOracle pipeline")
    parser.add_argument("--sample-size", type=int, default=None, help="Number of windows to sample for training (optional)")
    args = parser.parse_args()
    main(sample_size=args.sample_size)