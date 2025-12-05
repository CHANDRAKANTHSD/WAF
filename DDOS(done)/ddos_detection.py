import pandas as pd
import numpy as np
import time
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
import lightgbm as lgb
import xgboost as xgb
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import config

warnings.filterwarnings('ignore')

class DDoSDetector:
    def __init__(self):
        self.lgb_model = None
        self.xgb_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.selected_features = None
        self.best_model_name = None
        self.best_model = None
        
    def load_and_preprocess(self, file_path, dataset_name, sample_size=None):
        """Load and preprocess dataset"""
        print(f"\n{'='*80}")
        print(f"Loading {dataset_name}: {file_path}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # Load dataset with optional sampling for large files
        if sample_size:
            print(f"Sampling {sample_size} rows due to memory constraints...")
            # Read file size first
            total_rows = sum(1 for _ in open(file_path, encoding='latin1')) - 1
            skip_rows = sorted(np.random.choice(range(1, total_rows + 1), 
                                               total_rows - sample_size, 
                                               replace=False))
            df = pd.read_csv(file_path, encoding='latin1', low_memory=False, 
                           skiprows=skip_rows)
        else:
            df = pd.read_csv(file_path, encoding='latin1', low_memory=False)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()[:10]}...")  # Show first 10 columns
        
        # Handle different label column names
        label_col = None
        for col in [' Label', 'Label', 'label', 'type', 'Type']:
            if col in df.columns:
                label_col = col
                break
        
        if label_col is None:
            print("Available columns:", df.columns.tolist())
            raise ValueError("Label column not found")
        
        print(f"Label column: '{label_col}'")
        print(f"Label distribution:\n{df[label_col].value_counts()}")
        
        # Separate features and labels
        cols_to_drop = [label_col]
        # Also drop 'type' column if it exists (used for TON_IoT labeling)
        if 'type' in df.columns:
            cols_to_drop.append('type')
        
        X = df.drop(columns=cols_to_drop)
        y = df[label_col]
        
        # Convert labels to binary (DDoS vs Normal)
        # Check if there's a 'type' column (for TON_IoT dataset)
        if 'type' in df.columns:
            print(f"Using 'type' column for labeling")
            print(f"Type distribution:\n{df['type'].value_counts()}")
            # For TON_IoT: ddos and dos are DDoS attacks, others are not
            y_binary = df['type'].apply(lambda x: 1 if str(x).lower() in ['ddos', 'dos'] else 0)
        else:
            # For other datasets: check if label contains 'ddos' or 'dos'
            y_binary = y.apply(lambda x: 1 if 'ddos' in str(x).lower() or 'dos' in str(x).lower() else 0)
        
        print(f"Binary label distribution:\n{y_binary.value_counts()}")
        
        # Handle non-numeric columns
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                except:
                    X = X.drop(columns=[col])
        
        # Handle infinite and missing values
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)
        
        load_time = time.time() - start_time
        print(f"Preprocessing completed in {load_time:.2f} seconds")
        print(f"Final feature shape: {X.shape}")
        
        return X, y_binary
    
    def feature_selection(self, X, y, n_features=None):
        """Select top features"""
        if n_features is None:
            n_features = config.N_FEATURES
            
        print(f"\nSelecting top {n_features} features...")
        start_time = time.time()
        
        # Always select features fresh for each dataset to handle different column names
        self.feature_selector = SelectKBest(score_func=f_classif, k=min(n_features, X.shape[1]))
        X_selected = self.feature_selector.fit_transform(X, y)
        current_features = X.columns[self.feature_selector.get_support()].tolist()
        
        # Store features from first dataset only
        if self.selected_features is None:
            self.selected_features = current_features
        
        selection_time = time.time() - start_time
        print(f"Feature selection completed in {selection_time:.2f} seconds")
        print(f"Selected features: {current_features[:10]}...")
        print(f"Selected data shape: {X_selected.shape}")
        
        return X_selected
    
    def handle_imbalance(self, X, y):
        """Handle class imbalance using SMOTE"""
        print("\nHandling class imbalance with SMOTE...")
        start_time = time.time()
        
        print(f"Before SMOTE: {np.bincount(y)}")
        smote = SMOTE(random_state=42, k_neighbors=min(5, np.bincount(y).min() - 1))
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        balance_time = time.time() - start_time
        print(f"After SMOTE: {np.bincount(y_balanced)}")
        print(f"SMOTE completed in {balance_time:.2f} seconds")
        
        return X_balanced, y_balanced
    
    def train_models(self, X_train, X_test, y_train, y_test, dataset_name):
        """Train both LightGBM and XGBoost models"""
        print(f"\n{'='*80}")
        print(f"Training models on {dataset_name}")
        print(f"{'='*80}")
        
        results = {}
        
        # Train LightGBM
        print("\n--- Training LightGBM ---")
        lgb_start = time.time()
        
        if self.lgb_model is None:
            self.lgb_model = lgb.LGBMClassifier(**config.LIGHTGBM_PARAMS)
        
        self.lgb_model.fit(X_train, y_train)
        lgb_train_time = time.time() - lgb_start
        
        # LightGBM predictions
        lgb_pred_start = time.time()
        lgb_pred = self.lgb_model.predict(X_test)
        lgb_inference_time = (time.time() - lgb_pred_start) / len(X_test) * 1000  # ms per sample
        
        results['LightGBM'] = {
            'accuracy': accuracy_score(y_test, lgb_pred),
            'precision': precision_score(y_test, lgb_pred, zero_division=0),
            'recall': recall_score(y_test, lgb_pred, zero_division=0),
            'f1': f1_score(y_test, lgb_pred, zero_division=0),
            'train_time': lgb_train_time,
            'inference_time': lgb_inference_time
        }
        
        # Train XGBoost
        print("\n--- Training XGBoost ---")
        xgb_start = time.time()
        
        if self.xgb_model is None:
            self.xgb_model = xgb.XGBClassifier(**config.XGBOOST_PARAMS)
        
        self.xgb_model.fit(X_train, y_train)
        xgb_train_time = time.time() - xgb_start
        
        # XGBoost predictions
        xgb_pred_start = time.time()
        xgb_pred = self.xgb_model.predict(X_test)
        xgb_inference_time = (time.time() - xgb_pred_start) / len(X_test) * 1000  # ms per sample
        
        results['XGBoost'] = {
            'accuracy': accuracy_score(y_test, xgb_pred),
            'precision': precision_score(y_test, xgb_pred, zero_division=0),
            'recall': recall_score(y_test, xgb_pred, zero_division=0),
            'f1': f1_score(y_test, xgb_pred, zero_division=0),
            'train_time': xgb_train_time,
            'inference_time': xgb_inference_time
        }
        
        # Display results
        self.display_results(results, dataset_name)
        
        return results
    
    def display_results(self, results, dataset_name):
        """Display model comparison results"""
        print(f"\n{'='*80}")
        print(f"Results for {dataset_name}")
        print(f"{'='*80}")
        
        df_results = pd.DataFrame(results).T
        print(df_results.to_string())
        
        # Determine best model
        best_model = max(results.items(), key=lambda x: x[1]['f1'])
        print(f"\n🏆 Best Model: {best_model[0]} (F1-Score: {best_model[1]['f1']:.4f})")
        print(f"   Inference Time: {best_model[1]['inference_time']:.4f} ms/sample")
        
        return best_model[0]
    
    def plot_feature_importance(self, model_name, top_n=20):
        """Plot feature importance"""
        print(f"\nPlotting feature importance for {model_name}...")
        
        model = self.lgb_model if model_name == 'LightGBM' else self.xgb_model
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[-top_n:]
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(indices)), importances[indices])
            plt.yticks(range(len(indices)), [self.selected_features[i] for i in indices])
            plt.xlabel('Feature Importance')
            plt.title(f'{model_name} - Top {top_n} Feature Importances')
            plt.tight_layout()
            plt.savefig(f'{model_name}_feature_importance.png', dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved as '{model_name}_feature_importance.png'")
            plt.close()
    
    def save_models(self, model_name):
        """Save the best model"""
        print(f"\nSaving {model_name} model...")
        
        model = self.lgb_model if model_name == 'LightGBM' else self.xgb_model
        
        # Save as pickle
        with open(f'{model_name}_ddos_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved as '{model_name}_ddos_model.pkl'")
        
        # Save as joblib
        joblib.dump(model, f'{model_name}_ddos_model.joblib')
        print(f"Model saved as '{model_name}_ddos_model.joblib'")
        
        # Save scaler
        joblib.dump(self.scaler, 'scaler.joblib')
        print("Scaler saved as 'scaler.joblib'")
        
        # Save feature list
        with open('selected_features.pkl', 'wb') as f:
            pickle.dump(self.selected_features, f)
        print("Selected features saved as 'selected_features.pkl'")

def main():
    print("="*80)
    print("DDoS DETECTION SYSTEM - LightGBM vs XGBoost")
    print("="*80)
    
    # Initialize detector
    detector = DDoSDetector()
    
    # Load datasets from config
    datasets = [(d['path'], d['name']) for d in config.DATASETS]
    
    all_results = {}
    
    # Process each dataset sequentially
    for file_path, dataset_name in datasets:
        try:
            # Determine if we need to sample (for large datasets)
            sample_size = None
            if 'HOIC' in dataset_name or 'LOIC-HTTP' in dataset_name:
                sample_size = 500000  # Sample 500K rows for very large datasets
                print(f"⚠️  Large dataset detected - will sample {sample_size} rows")
            
            # Load and preprocess
            X, y = detector.load_and_preprocess(file_path, dataset_name, sample_size)
            
            # Check if we have both classes
            if len(np.unique(y)) < 2:
                print(f"⚠️  Skipping {dataset_name}: Only one class present (no DDoS samples)")
                continue
            
            # Feature selection
            X_selected = detector.feature_selection(X, y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=config.TEST_SIZE, 
                random_state=config.RANDOM_STATE, stratify=y
            )
            
            # Handle imbalance
            X_train_balanced, y_train_balanced = detector.handle_imbalance(X_train, y_train)
            
            # Normalize
            print("\nNormalizing features...")
            print(f"X_train_balanced shape before scaling: {X_train_balanced.shape}")
            print(f"X_test shape before scaling: {X_test.shape}")
            
            # Check if shapes match expected
            if X_train_balanced.shape[1] != 30 or X_test.shape[1] != 30:
                print(f"⚠️  Feature count mismatch! Expected 30, got train:{X_train_balanced.shape[1]}, test:{X_test.shape[1]}")
                print(f"⚠️  Skipping {dataset_name} due to feature mismatch")
                continue
            
            # Fit scaler only on first dataset, transform on subsequent ones
            if not hasattr(detector.scaler, 'n_features_in_'):
                X_train_scaled = detector.scaler.fit_transform(X_train_balanced)
                print(f"✓ Scaler fitted with {detector.scaler.n_features_in_} features")
            else:
                X_train_scaled = detector.scaler.transform(X_train_balanced)
                print(f"✓ Using existing scaler with {detector.scaler.n_features_in_} features")
            
            X_test_scaled = detector.scaler.transform(X_test)
            print(f"X_train_scaled shape: {X_train_scaled.shape}")
            print(f"X_test_scaled shape: {X_test_scaled.shape}")
            
            # Train and evaluate models
            results = detector.train_models(X_train_scaled, X_test_scaled, 
                                           y_train_balanced, y_test, dataset_name)
            
            all_results[dataset_name] = results
            
        except MemoryError as e:
            print(f"\n❌ Memory Error processing {dataset_name}: {str(e)}")
            print(f"   Try reducing sample_size or processing on a machine with more RAM")
            continue
        except Exception as e:
            print(f"\n❌ Error processing {dataset_name}: {str(e)}")
            import traceback
            print(f"   Details: {traceback.format_exc()[:200]}...")
            continue
    
    # Final comparison
    print(f"\n{'='*80}")
    print("FINAL MODEL COMPARISON ACROSS ALL DATASETS")
    print(f"{'='*80}")
    
    if not all_results:
        print("❌ No datasets were successfully processed!")
        print("   Please check the errors above and try again.")
        return
    
    print(f"\n✓ Successfully trained on {len(all_results)} dataset(s)")
    
    lgb_avg_f1 = np.mean([r['LightGBM']['f1'] for r in all_results.values()])
    xgb_avg_f1 = np.mean([r['XGBoost']['f1'] for r in all_results.values()])
    
    lgb_avg_inference = np.mean([r['LightGBM']['inference_time'] for r in all_results.values()])
    xgb_avg_inference = np.mean([r['XGBoost']['inference_time'] for r in all_results.values()])
    
    print(f"\nLightGBM - Average F1: {lgb_avg_f1:.4f}, Avg Inference: {lgb_avg_inference:.4f} ms/sample")
    print(f"XGBoost  - Average F1: {xgb_avg_f1:.4f}, Avg Inference: {xgb_avg_inference:.4f} ms/sample")
    
    # Select best model
    if lgb_avg_f1 > xgb_avg_f1:
        best_model_name = 'LightGBM'
        print(f"\n🏆 FINAL WINNER: LightGBM")
    else:
        best_model_name = 'XGBoost'
        print(f"\n🏆 FINAL WINNER: XGBoost")
    
    # Plot feature importance for best model
    detector.plot_feature_importance(best_model_name)
    
    # Save best model
    detector.save_models(best_model_name)
    
    print(f"\n{'='*80}")
    print("✅ Training completed successfully!")
    print(f"Best model ({best_model_name}) saved and ready for deployment")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
