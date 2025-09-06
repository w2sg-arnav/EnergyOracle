# =============================================================================
# Merged EnergyOracle for Kaggle Competition (Corrected Version)
#
# This script combines the best features of the "Enhanced" and "Quick" versions.
# - It uses the comprehensive feature engineering and robust pipeline from the
#   "EnhancedEnergyOracle".
# - It excludes the computationally expensive RandomForestRegressor to ensure
#   faster training, focusing on the high-performing XGBoost and LightGBM models.
# - The goal is to achieve a top-tier MSE score (< 0.005) while maintaining
#   a reasonable training time.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import Ridge, ElasticNet
import warnings
from tqdm import tqdm
import pickle
import os
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ==================== MERGED & OPTIMIZED VERSION ====================

class MergedEnergyOracle:
    """
    Production-ready EnergyOracle with competition-winning optimizations,
    excluding RandomForest for faster execution.
    """
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.models = {}
        self.building_models = {}  # For potential future building-specific models
        self.scalers = {}
        self.metadata_processed = None
        self.feature_importance = {}
        self.validation_scores = {}
        self.ensemble_weights = {}
        self.feature_cols = []
        
    def find_data_directory(self):
        """Find the correct data directory by searching common paths."""
        possible_paths = [
            "data/raw",
            "data", 
            ".",
            "./data",
            "./data/raw"
        ]
        
        for path in possible_paths:
            train_path = os.path.join(path, "train.csv")
            if os.path.exists(train_path):
                print(f"Found data in: {path}")
                return path
        
        # List current directory contents to help with debugging if data is not found
        print("Current directory contents:")
        for item in os.listdir("."):
            print(f"  {item}")
        
        raise FileNotFoundError("Could not find train.csv in any expected location. Please ensure data files are available.")
        
    def load_data(self):
        """Load all necessary data files from the found directory."""
        print("Loading data...")
        
        data_path = self.find_data_directory()
        
        train_df = pd.read_csv(os.path.join(data_path, "train.csv"), parse_dates=['timestamp'])
        test_df = pd.read_csv(os.path.join(data_path, "test.csv"), parse_dates=['timestamp'])
        metadata_df = pd.read_csv(os.path.join(data_path, "metadata.csv"))
        
        try:
            sample_submission = pd.read_csv(os.path.join(data_path, "sample_submission.csv"))
        except FileNotFoundError:
            print("sample_submission.csv not found, creating a placeholder.")
            sample_submission = pd.DataFrame({'row_id': [], 'meter_reading': []})
        
        print(f"Train shape: {train_df.shape}")
        print(f"Test shape: {test_df.shape}")
        print(f"Metadata shape: {metadata_df.shape}")
        
        return train_df, test_df, metadata_df, sample_submission
    
    def create_building_profiles(self, metadata_df):
        """Create comprehensive building profiles with advanced features."""
        print("Creating enhanced building profiles...")
        
        metadata_df = metadata_df.copy()
        
        # Advanced missing value handling using regional medians
        if 'region' in metadata_df.columns:
            metadata_df['area_in_sqft'] = metadata_df.groupby('region')['area_in_sqft'].transform(
                lambda x: x.fillna(x.median())
            )
        metadata_df['area_in_sqft'].fillna(metadata_df['area_in_sqft'].median(), inplace=True)
        
        # Fill other numeric missing values, assuming 0 for appliances not listed
        numeric_cols = ['no_of_people', 'rooms', 'lights', 'ceiling_fans', 'air_coolers', 
                       'air_conditioners', 'fridge', 'tv', 'water_heaters', 'washing_machine',
                       'mixer', 'iron', 'micro_wave', 'inverter']
        
        for col in numeric_cols:
            if col in metadata_df.columns:
                metadata_df[col].fillna(0, inplace=True)
        
        # Enhanced appliance weights based on typical consumption
        appliance_weights = {
            'air_conditioners': 3.5, 'water_heaters': 2.8, 'washing_machine': 1.5,
            'fridge': 1.0, 'micro_wave': 1.2, 'iron': 1.0, 'tv': 0.4,
            'air_coolers': 0.8, 'lights': 0.08, 'ceiling_fans': 0.1,
            'mixer': 0.3, 'inverter': 0.25
        }
        
        metadata_df['energy_intensity'] = 0
        for appliance, weight in appliance_weights.items():
            if appliance in metadata_df.columns:
                metadata_df['energy_intensity'] += metadata_df[appliance] * weight
        
        # Advanced derived features
        metadata_df['occupancy_density'] = metadata_df['no_of_people'] / (metadata_df['area_in_sqft'] + 1e-6)
        metadata_df['area_per_person'] = metadata_df['area_in_sqft'] / (metadata_df['no_of_people'] + 1e-6)
        metadata_df['appliance_density'] = metadata_df['energy_intensity'] / (metadata_df['area_in_sqft'] + 1e-6)
        metadata_df['rooms_per_person'] = metadata_df['rooms'] / (metadata_df['no_of_people'] + 1e-6)
        
        # Cooling/Heating load indicators
        metadata_df['cooling_load'] = (metadata_df['air_conditioners'] * 2.5 + 
                                     metadata_df['air_coolers'] * 0.6) / (metadata_df['area_in_sqft'] + 1e-6)
        metadata_df['heating_load'] = metadata_df['water_heaters'] / (metadata_df['no_of_people'] + 1e-6)
        
        # Granular building categories
        metadata_df['building_size'] = pd.cut(
            metadata_df['area_in_sqft'], 
            bins=[0, 500, 800, 1200, 1800, float('inf')],
            labels=['tiny', 'small', 'medium', 'large', 'xlarge']
        )
        
        # Appliance categories
        metadata_df['high_energy_appliances'] = (metadata_df['air_conditioners'] + 
                                               metadata_df['water_heaters'] + 
                                               metadata_df['washing_machine'])
        
        # Efficiency indicators and flags
        metadata_df['has_ac'] = (metadata_df['air_conditioners'] > 0).astype(int)
        metadata_df['has_heating'] = (metadata_df['water_heaters'] > 0).astype(int)
        
        metadata_df['efficiency_score'] = (metadata_df['energy_intensity'] / 
                                          (metadata_df['area_in_sqft'] * metadata_df['no_of_people'] + 1e-6))
        
        return metadata_df
    
    def create_temporal_features(self, df):
        """Create advanced temporal and cyclical features from the timestamp."""
        print("Creating advanced temporal features...")
        df = df.copy()
        
        ts = df['timestamp']
        df['hour'] = ts.dt.hour
        df['dayofweek'] = ts.dt.dayofweek
        df['month'] = ts.dt.month
        df['day_of_year'] = ts.dt.dayofyear
        df['week_of_year'] = ts.dt.isocalendar().week.astype(int)
        df['quarter'] = ts.dt.quarter
        
        # Cyclical feature encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Time-based flags for behavior patterns
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['is_peak_morning'] = df['hour'].isin([7, 8, 9]).astype(int)
        df['is_peak_evening'] = df['hour'].isin([18, 19, 20, 21]).astype(int)
        df['is_sleeping_hour'] = df['hour'].isin([0, 1, 2, 3, 4, 5]).astype(int)
        df['is_working_hour'] = df['hour'].isin(range(9, 18)).astype(int)
        
        # Seasonal patterns
        df['is_summer'] = df['month'].isin([4, 5, 6]).astype(int)
        df['is_winter'] = df['month'].isin([11, 12, 1]).astype(int)
        df['is_monsoon'] = df['month'].isin([7, 8, 9]).astype(int)
        
        # Interaction flags
        df['weekend_evening'] = df['is_weekend'] * df['is_peak_evening']
        
        return df
    
    def create_lag_and_rolling_features(self, df, target_col='meter_reading'):
        """Create optimized lag and rolling window features."""
        print("Creating optimized lag and rolling features...")
        df = df.copy().sort_values(['building_id', 'timestamp'])
        
        # Strategic lag features
        lag_hours = [1, 2, 3, 6, 12, 24, 48, 72, 168, 336] # Up to 2 weeks
        for lag in lag_hours:
            df[f'lag_{lag}h'] = df.groupby('building_id')[target_col].shift(lag)
        
        # Enhanced rolling window statistics
        for window in [6, 12, 24, 48, 168, 336]:
            grouped = df.groupby('building_id')[target_col]
            rolled = grouped.shift(1).rolling(window=window, min_periods=max(1, window // 4))
            
            df[f'rolling_mean_{window}h'] = rolled.mean()
            df[f'rolling_std_{window}h'] = rolled.std()
            df[f'rolling_min_{window}h'] = rolled.min()
            df[f'rolling_max_{window}h'] = rolled.max()
            df[f'rolling_median_{window}h'] = rolled.median()
        
        # Exponentially Weighted Mean (EWM)
        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
            df[f'ewm_{alpha}'] = df.groupby('building_id')[target_col].shift(1).ewm(alpha=alpha).mean()
        
        # Difference features to capture change
        df['diff_1h'] = df.groupby('building_id')[target_col].diff(1)
        df['diff_24h'] = df.groupby('building_id')[target_col].diff(24)
        
        # Rate of change
        df['rate_1h'] = df['diff_1h'] / (df['lag_1h'] + 1e-6)
        
        return df
    
    def create_advanced_features(self, df, metadata_df):
        """Create advanced interaction and contextual features by merging dataframes."""
        print("Creating advanced interaction features...")
        
        df = df.merge(metadata_df, on='building_id', how='left')
        
        # Hour-appliance interactions
        df['ac_cooling_demand'] = (df['air_conditioners'] * 
                                  (df['is_peak_evening'] + df['hour'].apply(lambda x: max(0, x-12)/12)) * 
                                  df['is_summer'])
        
        df['heating_demand'] = (df['water_heaters'] * 
                              (df['is_peak_morning'] + df['is_peak_evening']) * 
                              (df['is_winter'] + 0.3))
        
        df['occupancy_pattern'] = (df['occupancy_density'] * 
                                 (df['is_peak_morning'] + df['is_peak_evening'] + df['weekend_evening']))
        
        # Seasonal building interactions
        # === THIS IS THE CORRECTED LINE ===
        df['summer_ac_load'] = df['is_summer'] * df['cooling_load'] * ((df['hour'] > 10) & (df['hour'] < 22))
        df['winter_heating_load'] = df['is_winter'] * df['heating_load'] * df['is_peak_morning']
        
        # Get dummies for categorical building sizes
        building_dummies = pd.get_dummies(df['building_size'], prefix='size')
        df = pd.concat([df, building_dummies], axis=1)
        
        # Advanced ratio features
        df['people_per_room'] = df['no_of_people'] / (df['rooms'] + 1e-6)
        df['appliance_per_person'] = df['energy_intensity'] / (df['no_of_people'] + 1e-6)
        
        return df
    
    def prepare_features(self, df, feature_cols):
        """Prepare the feature matrix with robust preprocessing."""
        available_cols = [col for col in feature_cols if col in df.columns]
        X = df[available_cols].copy()
        
        # Robust missing value handling
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        X.replace([np.inf, -np.inf], 0, inplace=True)
        
        # Outlier clipping at the 99.5th percentile to improve model stability
        for col in X.select_dtypes(include=[np.number]).columns:
            upper_limit = X[col].quantile(0.995)
            lower_limit = X[col].quantile(0.005)
            X[col] = X[col].clip(lower_limit, upper_limit)
        
        return X, available_cols
    
    def create_optimized_models(self):
        """Create an optimized ensemble of gradient boosting and linear models."""
        models = {
            'xgb_main': xgb.XGBRegressor(
                n_estimators=1500,
                max_depth=7,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                early_stopping_rounds=80,
                n_jobs=-1,
                tree_method='hist'
            ),
            'lgb_main': lgb.LGBMRegressor(
                n_estimators=1500,
                max_depth=7,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbose=-1,
                n_jobs=-1,
                force_col_wise=True
            ),
            'xgb_deep': xgb.XGBRegressor(
                n_estimators=1200,
                max_depth=10,
                learning_rate=0.02,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=123,
                early_stopping_rounds=80,
                n_jobs=-1,
                tree_method='hist'
            ),
            'elastic': ElasticNet(
                alpha=0.01,
                l1_ratio=0.5,
                random_state=42,
                max_iter=2000
            )
        }
        return models
    
    def train_models(self, train_df, metadata_df, validation_split=0.15):
        """Train the full ensemble of models."""
        print("Starting model training process...")
        
        # Feature engineering pipeline
        self.metadata_processed = self.create_building_profiles(metadata_df)
        train_processed = self.create_temporal_features(train_df)
        train_processed = self.create_lag_and_rolling_features(train_processed)
        train_processed = self.create_advanced_features(train_processed, self.metadata_processed)
        
        input_data = train_processed[train_processed['role'] == 'input'].copy()
        
        # Define comprehensive feature set
        self.feature_cols = [col for col in train_processed.columns if col not in [
            'timestamp', 'meter_reading', 'building_id', 'role', 'row_id',
            'building_size', 'region' # Base columns, not direct features
        ]]
        
        # Prepare data
        X, self.feature_cols = self.prepare_features(input_data, self.feature_cols)
        y = input_data['meter_reading'].values
        
        print(f"Feature matrix shape: {X.shape}")
        
        valid_idx = ~np.isnan(y) & np.isfinite(y) & (y >= 0)
        X, y = X[valid_idx], y[valid_idx]
        
        # Time-based split for validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
        
        models = self.create_optimized_models()
        
        for name, model in models.items():
            print(f"Training {name}...")
            try:
                if 'xgb' in name:
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                elif 'lgb' in name:
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                              callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
                else:
                    model.fit(X_train, y_train)
                
                val_pred = model.predict(X_val)
                val_mse = mean_squared_error(y_val, val_pred)
                print(f"  ✓ {name} validation MSE: {val_mse:.6f}")
                
                self.validation_scores[name] = val_mse
                self.models[name] = model
                
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = pd.DataFrame({
                        'feature': self.feature_cols,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
            except Exception as e:
                print(f"  ✗ Error training {name}: {e}")
        
        self.calculate_ensemble_weights()
        return self.models
    
    def calculate_ensemble_weights(self):
        """Calculate optimal ensemble weights using inverse MSE."""
        if not self.validation_scores:
            self.ensemble_weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
            return
        
        inv_scores = {name: 1.0 / (score + 1e-6) for name, score in self.validation_scores.items()}
        total_inv = sum(inv_scores.values())
        
        self.ensemble_weights = {name: inv_score / total_inv for name, inv_score in inv_scores.items()}
        
        print("\nCalculated Ensemble Weights:")
        for name, weight in self.ensemble_weights.items():
            print(f"  - {name}: {weight:.3f}")
    
    def predict_ensemble(self, X):
        """Make predictions using the weighted ensemble."""
        predictions = []
        
        for name, model in self.models.items():
            weight = self.ensemble_weights.get(name, 1.0 / len(self.models))
            pred = model.predict(X)
            predictions.append(pred * weight)
        
        ensemble_pred = np.sum(predictions, axis=0)
        return np.maximum(ensemble_pred, 0) # Ensure non-negativity
    
    def apply_post_processing(self, predictions, building_data):
        """Apply intelligent post-processing constraints to predictions."""
        # Dynamic upper bound based on building characteristics
        energy_intensity = building_data['energy_intensity'].values
        area = building_data['area_in_sqft'].values
        max_reasonable = energy_intensity * 8 + area * 0.02
        
        # Apply constraints
        predictions = np.maximum(predictions, 0)
        predictions = np.minimum(predictions, max_reasonable)
        predictions = np.minimum(predictions, 25) # Global cap
        
        return predictions
    
    def generate_submission(self, test_df, metadata_df):
        """Generate the final submission file."""
        print("Generating submission file...")
        
        # Feature engineering on test data
        test_processed = self.create_temporal_features(test_df)
        test_processed = self.create_lag_and_rolling_features(test_processed)
        test_processed = self.create_advanced_features(test_processed, self.metadata_processed)
        
        target_rows = test_processed[test_processed['role'] == 'target'].copy()
        
        X_test, _ = self.prepare_features(target_rows, self.feature_cols)
        
        predictions = self.predict_ensemble(X_test)
        predictions = self.apply_post_processing(predictions, target_rows)
        
        submission = pd.DataFrame({
            'row_id': target_rows['row_id'].values,
            'meter_reading': predictions
        })
        
        print(f"Submission shape: {submission.shape}")
        print(f"Prediction stats: min={predictions.min():.4f}, max={predictions.max():.4f}, mean={predictions.mean():.4f}")
        
        return submission

# ==================== MAIN EXECUTION & UTILITIES ====================

def create_submission_variants(oracle, test_df, metadata_df, timestamp):
    """Create submission variants for ensemble diversification."""
    print("Creating submission variants...")
    
    try:
        # Generate base predictions
        test_processed = oracle.create_temporal_features(test_df)
        test_processed = oracle.create_lag_and_rolling_features(test_processed)
        test_processed = oracle.create_advanced_features(test_processed, oracle.metadata_processed)
        target_rows = test_processed[test_processed['role'] == 'target'].copy()
        X_test, _ = oracle.prepare_features(target_rows, oracle.feature_cols)
        
        # Variant 1: Conservative (more regularized linear model influence)
        # Weights are adjusted since RandomForest was removed
        conservative_weights = {'xgb_main': 0.35, 'lgb_main': 0.35, 'elastic': 0.30}
        
        conservative_preds = []
        for name, model in oracle.models.items():
            if name in conservative_weights:
                pred = model.predict(X_test) * conservative_weights[name]
                conservative_preds.append(pred)
        
        if conservative_preds:
            final_pred = np.sum(conservative_preds, axis=0)
            final_pred = oracle.apply_post_processing(final_pred, target_rows)
            submission = pd.DataFrame({'row_id': target_rows['row_id'].values, 'meter_reading': final_pred})
            submission.to_csv(f'conservative_submission_{timestamp}.csv', index=False)
            print("  ✓ Conservative variant created")
        
        # Variant 2: Aggressive (pure gradient boosting)
        if 'xgb_main' in oracle.models and 'lgb_main' in oracle.models:
            aggressive_pred = (oracle.models['xgb_main'].predict(X_test) * 0.5 + 
                             oracle.models['lgb_main'].predict(X_test) * 0.5)
            aggressive_pred = oracle.apply_post_processing(aggressive_pred, target_rows)
            submission = pd.DataFrame({'row_id': target_rows['row_id'].values, 'meter_reading': aggressive_pred})
            submission.to_csv(f'aggressive_submission_{timestamp}.csv', index=False)
            print("  ✓ Aggressive variant created")
            
    except Exception as e:
        print(f"  ✗ Warning: Could not create variants - {e}")

def display_feature_importance(oracle):
    """Display feature importance from all trained models."""
    print("\n" + "="*50)
    print("Feature Importance Analysis")
    print("="*50)
    
    if not oracle.feature_importance:
        print("No feature importance data available.")
        return
    
    for model_name, importance_df in oracle.feature_importance.items():
        print(f"\n{model_name.upper()} - Top 15 Features:")
        print(importance_df.head(15).to_string(index=False, float_format='%.4f'))

def generate_competition_insights(oracle, submission):
    """Generate insights and strategic recommendations for the competition."""
    print("\n" + "="*50)
    print("COMPETITION INSIGHTS & STRATEGY")
    print("="*50)
    
    preds = submission['meter_reading'].values
    
    print(f"📈 Prediction Statistics:")
    print(f"   Mean: {preds.mean():.4f}, Std: {preds.std():.4f}, Min: {preds.min():.4f}, Max: {preds.max():.4f}")
    
    print(f"\n🎯 Model Performance Summary:")
    if oracle.validation_scores:
        best_model = min(oracle.validation_scores.items(), key=lambda x: x[1])
        weighted_score = sum(s * oracle.ensemble_weights.get(n, 0) for n, s in oracle.validation_scores.items())
        print(f"   Best single model: {best_model[0]} (MSE: {best_model[1]:.6f})")
        print(f"   Weighted Ensemble Validation MSE: {weighted_score:.6f}")
        
        if weighted_score < 0.005:
            print("   🚀 EXCELLENT: Performance is within the target winning range.")
        elif weighted_score < 0.01:
            print("   👍 GOOD: Strong competitive performance.")
        else:
            print("   ⚠️  NEEDS REVIEW: Performance may be below the top tier.")
            
    print(f"\n📁 Submission Recommendations:")
    print("   1. Submit the main `merged_submission.csv` first.")
    print("   2. Use `conservative_submission.csv` if the main one appears to overfit the public leaderboard.")
    print("   3. Use `aggressive_submission.csv` for a high-risk, high-reward attempt.")

def main():
    """Main execution function to run the entire pipeline."""
    print("=== Merged EnergyOracle Competition Pipeline ===")
    
    oracle = MergedEnergyOracle(data_dir=".")
    
    try:
        # Load and validate data
        train_df, test_df, metadata_df, _ = oracle.load_data()
        print(f"\nTrain date range: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
        print(f"Test date range: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
        
        # Train models
        print("\n" + "="*50)
        models = oracle.train_models(train_df, metadata_df, validation_split=0.15)
        
        if not models:
            raise RuntimeError("ERROR: No models were trained successfully!")
        
        # Generate submission
        print("\n" + "="*50)
        submission = oracle.generate_submission(test_df, metadata_df)
        
        # Save files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        main_filename = f'merged_submission_{timestamp}.csv'
        submission.to_csv(main_filename, index=False)
        print(f"\nMain submission saved: {main_filename}")
        
        submission.to_csv('merged_energyoracle_submission.csv', index=False)
        print("Backup submission saved: merged_energyoracle_submission.csv")
        
        # Create and save variants
        create_submission_variants(oracle, test_df, metadata_df, timestamp)
        
        # Display insights
        display_feature_importance(oracle)
        generate_competition_insights(oracle, submission)
        
        print(f"\n{'='*50}")
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("🚀 Ready for Kaggle submission!")
        
    except Exception as e:
        print(f"\nERROR in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()