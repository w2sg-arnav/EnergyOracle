import pandas as pd
import numpy as np
from tqdm import tqdm
from src import config
from src.feature_engineering import create_temporal_features, create_lag_features, create_metadata_features

def make_predictions(test_df, metadata_df, model, features):
    """
    Generate predictions for the test set and create submission file.
    """
    print("--- Generating Submission ---")
    test_full = pd.merge(test_df, create_metadata_features(metadata_df), on='building_id', how='left')
    test_full = create_temporal_features(test_full)
    test_full = create_lag_features(test_full, config.LAG_FEATURES)
    
    submission = []
    window_ids = test_full['window_id'].unique()
    print(f"Processing {len(window_ids)} windows")
    
    for window_id in tqdm(window_ids, desc="Predicting"):
        window_data = test_full[test_full['window_id'] == window_id]
        input_data = window_data[window_data['role'] == 'input']
        target_rows = window_data[window_data['role'] == 'target']
        if len(input_data) != 168 or len(target_rows) != 24:
            print(f"Skipping window_id {window_id}: invalid input ({len(input_data)}) or target ({len(target_rows)}) length")
            continue
        
        # Extract aggregate features
        agg_features = input_data.groupby('window_id').agg({
            'meter_reading': ['mean', 'std', 'min', 'max']
        })
        agg_features.columns = ['meter_' + '_'.join(col).strip() for col in agg_features.columns.values]
        agg_features = agg_features.reset_index()
        
        # Extract recent features
        recent_features = input_data.tail(1).reset_index()
        recent_cols = [
            'lag_1', 'lag_48', 'lag_168',
            'rolling_mean_6', 'rolling_std_6', 'rolling_min_6', 'rolling_max_6',
            'rolling_mean_12', 'rolling_std_12', 'rolling_min_12', 'rolling_max_12',
            'rolling_mean_24', 'rolling_std_24', 'rolling_min_24', 'rolling_max_24',
            'rolling_mean_168', 'rolling_std_168', 'rolling_min_168', 'rolling_max_168',
            'lag_diff_1_24',
            'energy_hour_interaction', 'ac_hour_interaction',
            'occupancy_density', 'energy_intensity', 'total_appliances', 
            'ac_occupancy_interaction', 'energy_intensity_per_person'
        ]
        recent_features = recent_features[recent_cols + ['window_id']]
        
        # Merge features
        features_df = pd.merge(agg_features, recent_features, on='window_id', how='left')
        
        # Add target hour features
        features_df = features_df.loc[features_df.index.repeat(24)].reset_index(drop=True)
        features_df['target_hour'] = np.tile(np.arange(24), len(features_df) // 24)
        features_df['target_day_of_week'] = target_rows['timestamp'].dt.dayofweek.values[0]
        features_df['temp_day_interaction'] = features_df['target_hour'] * features_df['target_day_of_week']
        
        # Debug feature columns
        missing_features = [f for f in features if f not in features_df.columns]
        if missing_features:
            print(f"Feature mismatch for window_id {window_id}: {missing_features} not in features_df")
            continue
        
        features_df = features_df[features]
        
        # Check for NaNs
        if features_df[features].isnull().any().any():
            print(f"Warning: NaNs detected in features for window_id {window_id}")
            print(f"NaNs in features_df: {features_df[features].isnull().sum()}")
            features_df = features_df.fillna(features_df.mean(numeric_only=True))
        
        # Predict
        pred_log = model.predict(features_df)
        pred = np.expm1(pred_log).flatten()
        
        # Post-process
        max_capacity = features_df['energy_intensity'].iloc[0] * 2
        pred = np.clip(pred, 0, max_capacity)
        
        # Append to submission
        for row_id, pred_value in zip(target_rows['row_id'], pred):
            submission.append({'row_id': row_id, 'meter_reading': pred_value})
        
        # Log progress every 10%
        if window_id % (len(window_ids) // 10) == 0:
            print(f"Progress: {window_id / len(window_ids) * 100:.0f}%")
    
    submission_df = pd.DataFrame(submission)
    submission_df.to_csv(config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {config.SUBMISSION_FILE}")
    print(f"Submission shape: {submission_df.shape}")
    return submission_df