import pandas as pd
import numpy as np

def create_metadata_features(metadata_df):
    print("Creating metadata features...")
    df = metadata_df.copy()
    
    # Handle missing values
    print("Handling missing values in metadata...")
    df.fillna({
        'no_of_people': df['no_of_people'].median(),
        'area_in_sqft': df['area_in_sqft'].median(),
        'inverter': 0,
        'lights': df['lights'].median(),
        'ceiling_fans': df['ceiling_fans'].median(),
        'air_coolers': 0,
        'air_conditioners': 0,
        'fridge': 0,
        'tv': 0,
        'water_heaters': 0,
        'washing_machine': 0,
        'mixer': 0,
        'iron': 0,
        'micro_wave': 0
    }, inplace=True)
    
    # Create derived features
    df['occupancy_density'] = df['no_of_people'] / df['area_in_sqft']
    df['total_appliances'] = df[['inverter', 'lights', 'ceiling_fans', 'air_coolers', 
                                 'air_conditioners', 'fridge', 'tv', 'water_heaters', 
                                 'washing_machine', 'mixer', 'iron', 'micro_wave']].sum(axis=1)
    df['energy_intensity'] = df['total_appliances'] / df['area_in_sqft']
    df['ac_occupancy_interaction'] = df['air_conditioners'] * df['occupancy_density']
    df['energy_intensity_per_person'] = df['energy_intensity'] / (df['no_of_people'] + 1e-6)  # Avoid division by zero
    
    # One-hot encode region
    df = pd.get_dummies(df, columns=['region'], prefix='region')
    
    print(f"Metadata features created. Shape: {df.shape}")
    print(f"Missing values after processing: {df.isnull().sum().sum()}")
    
    return df

def create_temporal_features(df):
    print("Creating temporal features...")
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # New feature
    df['energy_day_interaction'] = df['energy_intensity'] * df['day_of_week_sin']
    
    print(f"Temporal features created. Shape: {df.shape}")
    return df

def create_lag_features(df, lag_features):
    print("Creating lag and rolling features...")
    df = df.copy()
    df = df.sort_values(by=['building_id', 'timestamp'])
    
    for lag in lag_features:
        df[f'lag_{lag}'] = df.groupby('building_id')['meter_reading'].shift(lag)
    
    for window in [6, 12, 24, 168]:
        df[f'rolling_mean_{window}'] = df.groupby('building_id')['meter_reading'].shift(1).rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df.groupby('building_id')['meter_reading'].shift(1).rolling(window=window).std()
        df[f'rolling_min_{window}'] = df.groupby('building_id')['meter_reading'].shift(1).rolling(window=window).min()
        df[f'rolling_max_{window}'] = df.groupby('building_id')['meter_reading'].shift(1).rolling(window=window).max()
    
    # Additional interaction features
    df['lag_diff_1_24'] = df['lag_1'] - df.get('lag_24', 0)  # Handle missing lag_24
    df['energy_hour_interaction'] = df['energy_intensity'] * df['hour_sin']
    df['ac_hour_interaction'] = df['air_conditioners'] * df['hour_sin']
    
    df = df.bfill().ffill()
    print(f"Lag and rolling features created. Shape: {df.shape}")
    return df