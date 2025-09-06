import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from src import config
from src.feature_engineering import create_metadata_features, create_temporal_features, create_lag_features

def train_model(train_df, metadata_df, sample_size=None, use_lightgbm=False):
    """
    Train a multi-output model for 24-hour forecasting.
    
    Args:
        train_df: Training DataFrame
        metadata_df: Metadata DataFrame
        sample_size: Number of windows to sample for debugging (optional)
        use_lightgbm: Use LightGBM instead of XGBoost (default: False)
    """
    print("--- Preparing Training Data ---")
    # Merge metadata
    processed_metadata_df = create_metadata_features(metadata_df)
    train_full = pd.merge(train_df, processed_metadata_df, on='building_id', how='left')
    train_full = create_temporal_features(train_full)
    train_full = create_lag_features(train_full, config.LAG_FEATURES)
    
    # Debug available columns
    print(f"Columns in train_full after feature engineering: {list(train_full.columns)}")
    
    # Sample data for debugging
    if sample_size:
        window_ids = train_df['window_id'].unique()[:sample_size]
        train_full = train_full[train_full['window_id'].isin(window_ids)]
        print(f"Using sample of {sample_size} windows")
    
    # Separate inputs and targets
    inputs = train_full[train_full['role'] == 'input'].copy().sort_values(by=['building_id', 'timestamp'])
    targets = train_full[train_full['role'] == 'target'].copy()
    
    # Add target-specific temporal features
    targets['target_hour'] = targets['timestamp'].dt.hour
    targets['target_day_of_week'] = targets['timestamp'].dt.dayofweek
    targets['target_hour_sin'] = np.sin(2 * np.pi * targets['target_hour'] / 24)
    targets['target_hour_cos'] = np.cos(2 * np.pi * targets['target_hour'] / 24)
    
    # Check for duplicates in target data
    print("--- Checking for duplicates in target data ---")
    duplicates = targets[targets.duplicated(subset=['window_id', 'target_hour'], keep=False)]
    if not duplicates.empty:
        print(f"Found {len(duplicates)} duplicate rows for window_id and target_hour:")
        print(duplicates[['window_id', 'target_hour', 'meter_reading']].head())
        print("Removing duplicates by keeping first occurrence...")
        targets = targets.drop_duplicates(subset=['window_id', 'target_hour'], keep='first')
    
    # Pivot targets to have 24 columns (one per hour)
    try:
        target_pivot = targets.pivot(index='window_id', columns='target_hour', values='meter_reading')
        target_pivot.columns = [f'hour_{h}' for h in target_pivot.columns]
    except ValueError as e:
        print(f"Error during pivot: {e}")
        raise
    
    # Aggregate input features
    agg_features = inputs.groupby('window_id').agg({
        'meter_reading': ['mean', 'std', 'min', 'max']
    })
    agg_features.columns = ['meter_' + '_'.join(col).strip() for col in agg_features.columns.values]
    agg_features = agg_features.reset_index()
    
    # Get most recent input features
    recent_features = inputs.groupby('window_id').last().reset_index()
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
    
    # Merge features
    final_df = pd.merge(target_pivot, agg_features, on='window_id', how='left')
    final_df = pd.merge(final_df, recent_features[recent_cols + ['window_id']], on='window_id', how='left')
    
    # Add target hour features and new interaction for each row before selection
    final_df = final_df.loc[final_df.index.repeat(24)].reset_index(drop=True)
    final_df['target_hour'] = np.tile(np.arange(24), len(final_df) // 24)
    final_df['target_day_of_week'] = targets.groupby('window_id')['target_day_of_week'].first().reindex(final_df['window_id']).values
    final_df['temp_day_interaction'] = final_df['target_hour'] * final_df['target_day_of_week']
    
    # Define features and targets
    features = recent_cols + [
        'meter_meter_reading_mean', 'meter_meter_reading_std', 
        'meter_meter_reading_min', 'meter_meter_reading_max',
        'target_day_of_week', 'temp_day_interaction'  # Removed target_hour
    ]
    target_cols = [f'hour_{h}' for h in range(24)]
    
    # Debug feature availability
    print(f"Columns in final_df before feature selection: {list(final_df.columns)}")
    missing_features = [f for f in features if f not in final_df.columns]
    if missing_features:
        print(f"Error: Missing features in final_df: {missing_features}")
        raise KeyError(f"Missing features: {missing_features}")
    
    # Drop NaNs
    final_df.dropna(subset=features + target_cols, inplace=True)
    
    X = final_df[features]
    y = final_df[target_cols]
    
    # Time-based split
    unique_windows = final_df['window_id'].unique()
    train_windows, val_windows = train_test_split(unique_windows, test_size=0.2, shuffle=False)
    train_idx = final_df['window_id'].isin(train_windows)
    val_idx = final_df['window_id'].isin(val_windows)
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Log-transform targets
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    
    # Debug shapes
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train_log shape: {y_train_log.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val_log shape: {y_val_log.shape}")
    print(f"Features used: {features}")
    
    # Train multi-output model
    print("--- Training Multi-Output Model ---")
    model_class = LGBMRegressor if use_lightgbm else xgb.XGBRegressor
    model = MultiOutputRegressor(model_class(**config.MODEL_PARAMS))
    
    # Fit each estimator separately
    estimators = []
    for i in range(y_train_log.shape[1]):
        print(f"Training model for hour {i}")
        estimator = model_class(**config.MODEL_PARAMS)
        estimator.fit(
            X_train,
            y_train_log.iloc[:, i],
            eval_set=[(X_val, y_val_log.iloc[:, i])],
            verbose=100
        )
        estimators.append(estimator)
    
    model.estimators_ = estimators
    
    # Evaluate
    val_preds_log = model.predict(X_val)
    val_preds = np.expm1(val_preds_log)
    mse = mean_squared_error(y_val, val_preds)
    rmse = np.sqrt(mse)
    print(f"Validation MSE: {mse:.4f}")
    print(f"Validation RMSE: {rmse:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
    })
    print("Feature Importance:")
    print(feature_importance.sort_values('importance', ascending=False))
    
    # Save plot
    import matplotlib.pyplot as plt
    plt.plot(y_val.iloc[0], label='Actual')
    plt.plot(val_preds[0], label='Predicted')
    plt.legend()
    plt.title("Predictions vs Actuals (First Window)")
    plt.savefig(f'{config.MODELS_DIR}/prediction_vs_actual.png')
    plt.close()
    
    # Save model
    import pickle
    model_path = f'{config.MODELS_DIR}/{"lightgbm" if use_lightgbm else "xgboost"}_multioutput.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")
    
    return model, features