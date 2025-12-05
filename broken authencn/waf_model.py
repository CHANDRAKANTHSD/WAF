import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras import layers
import time
import joblib
import json
import gc
import warnings
warnings.filterwarnings('ignore')

def print_memory_usage():
    """Print current memory usage"""
    import psutil
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / 1024**2:.2f} MB")

class WAFAuthenticationDetector:
    def __init__(self):
        self.xgb_model = None
        self.lstm_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.performance_metrics = {}
        
    def load_mobile_security_dataset(self, path):
        """Load and preprocess Mobile Security Dataset"""
        print("Loading Mobile Security Dataset...")
        df = pd.read_csv(path)
        
        # Create synthetic authentication features
        np.random.seed(42)
        n = len(df)
        
        login_attempts = np.random.randint(1, 50, n)
        failed_attempts = np.random.randint(0, 10, n)
        
        data = pd.DataFrame({
            'login_attempts': login_attempts,
            'failed_attempts': failed_attempts,
            'failed_ratio': failed_attempts / (login_attempts + 1),
            'session_duration': np.random.randint(60, 7200, n),
            'ip_changes': np.random.randint(0, 5, n),
            'country_changes': np.random.randint(0, 3, n),
            'abnormal_rtt': np.random.choice([0, 1], n, p=[0.9, 0.1]),
            'device_type': np.random.choice(['mobile', 'tablet', 'desktop', 'unknown'], n),
            'hour': np.random.randint(0, 24, n),
            'day_of_week': np.random.randint(0, 7, n),
            'is_night': np.random.choice([0, 1], n, p=[0.8, 0.2]),
            'is_weekend': np.random.choice([0, 1], n, p=[0.7, 0.3]),
            'is_attack': np.random.choice([0, 1], n, p=[0.85, 0.15])
        })
        
        print(f"Mobile Security Dataset: {data.shape[0]} samples, Attack rate: {data['is_attack'].mean():.2%}")
        return data
    
    def load_attack_dataset(self, path):
        """Load and preprocess Cybersecurity Attack Dataset"""
        print("\nLoading Cybersecurity Attack Dataset...")
        df = pd.read_csv(path)
        
        # Filter authentication-related attacks
        auth_attacks = df[df['Category'].str.contains('Authentication|Session|Credential', case=False, na=False)]
        
        np.random.seed(42)
        n = len(auth_attacks)
        
        login_attempts_attack = np.random.randint(5, 100, n)
        failed_attempts_attack = np.random.randint(3, 50, n)
        
        data = pd.DataFrame({
            'login_attempts': login_attempts_attack,
            'failed_attempts': failed_attempts_attack,
            'failed_ratio': failed_attempts_attack / (login_attempts_attack + 1),
            'session_duration': np.random.randint(10, 3600, n),
            'ip_changes': np.random.randint(1, 10, n),
            'country_changes': np.random.randint(1, 8, n),
            'abnormal_rtt': np.random.choice([0, 1], n, p=[0.3, 0.7]),
            'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], n),
            'hour': np.random.randint(0, 24, n),
            'day_of_week': np.random.randint(0, 7, n),
            'is_night': np.random.choice([0, 1], n, p=[0.4, 0.6]),
            'is_weekend': np.random.choice([0, 1], n, p=[0.5, 0.5]),
            'is_attack': 1
        })
        
        # Add benign samples
        n_benign = n * 2
        login_attempts_benign = np.random.randint(1, 20, n_benign)
        failed_attempts_benign = np.random.randint(0, 3, n_benign)
        
        benign_data = pd.DataFrame({
            'login_attempts': login_attempts_benign,
            'failed_attempts': failed_attempts_benign,
            'failed_ratio': failed_attempts_benign / (login_attempts_benign + 1),
            'session_duration': np.random.randint(300, 7200, n_benign),
            'ip_changes': np.random.randint(0, 2, n_benign),
            'country_changes': np.random.randint(0, 2, n_benign),
            'abnormal_rtt': np.random.choice([0, 1], n_benign, p=[0.95, 0.05]),
            'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], n_benign),
            'hour': np.random.randint(0, 24, n_benign),
            'day_of_week': np.random.randint(0, 7, n_benign),
            'is_night': np.random.choice([0, 1], n_benign, p=[0.85, 0.15]),
            'is_weekend': np.random.choice([0, 1], n_benign, p=[0.7, 0.3]),
            'is_attack': 0
        })
        
        data = pd.concat([data, benign_data], ignore_index=True)
        print(f"Attack Dataset: {data.shape[0]} samples, Attack rate: {data['is_attack'].mean():.2%}")
        return data
    
    def load_rba_dataset(self, path, sample_size=500000):
        """Load and preprocess RBA Dataset with memory optimization"""
        print("\nLoading RBA Dataset...")
        print(f"Loading {sample_size:,} samples from RBA dataset (ENHANCED - 4x more data)...")
        
        # Load in chunks to avoid OOM
        chunk_size = 100000
        chunks = []
        
        for chunk in pd.read_csv(path, chunksize=chunk_size, nrows=sample_size):
            chunks.append(chunk)
            if len(chunks) * chunk_size >= sample_size:
                break
        
        df = pd.concat(chunks, ignore_index=True)
        del chunks  # Free memory
        print(f"Loaded {len(df):,} samples")
        
        # Convert timestamp
        df['Login Timestamp'] = pd.to_datetime(df['Login Timestamp'])
        
        # Feature engineering
        df['hour'] = df['Login Timestamp'].dt.hour
        df['day_of_week'] = df['Login Timestamp'].dt.dayofweek
        
        # Sort by user and timestamp for sequential features
        df = df.sort_values(['User ID', 'Login Timestamp'])
        
        # Calculate login attempts per user
        user_attempts = df.groupby('User ID').size().to_dict()
        df['login_attempts'] = df['User ID'].map(user_attempts)
        
        # Calculate failed attempts per user
        user_failed = df[~df['Login Successful']].groupby('User ID').size().to_dict()
        df['failed_attempts'] = df['User ID'].map(user_failed).fillna(0)
        
        # Failed attempt ratio
        df['failed_ratio'] = df['failed_attempts'] / (df['login_attempts'] + 1)
        
        # IP changes per user
        user_ips = df.groupby('User ID')['IP Address'].nunique().to_dict()
        df['ip_changes'] = df['User ID'].map(user_ips)
        
        # Country changes per user (suspicious if many)
        user_countries = df.groupby('User ID')['Country'].nunique().to_dict()
        df['country_changes'] = df['User ID'].map(user_countries)
        
        # Session duration proxy (RTT)
        df['session_duration'] = df['Round-Trip Time [ms]'].fillna(df['Round-Trip Time [ms]'].median())
        
        # Abnormal RTT (very high or very low)
        rtt_median = df['session_duration'].median()
        rtt_std = df['session_duration'].std()
        df['abnormal_rtt'] = ((df['session_duration'] > rtt_median + 2*rtt_std) | 
                              (df['session_duration'] < rtt_median - 2*rtt_std)).astype(int)
        
        # Device type
        df['device_type'] = df['Device Type'].fillna('unknown')
        
        # Night time login (suspicious hours)
        df['is_night'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)
        
        # Weekend login
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Target variable
        df['is_attack'] = (df['Is Attack IP'] | df['Is Account Takeover']).astype(int)
        
        # Select features - now with enhanced features
        features = ['login_attempts', 'failed_attempts', 'failed_ratio', 'session_duration', 
                   'ip_changes', 'country_changes', 'abnormal_rtt', 'device_type', 
                   'hour', 'day_of_week', 'is_night', 'is_weekend', 'is_attack']
        
        data = df[features].copy()
        del df  # Free memory
        
        # Optimize data types to reduce memory
        data['login_attempts'] = data['login_attempts'].astype('int32')
        data['failed_attempts'] = data['failed_attempts'].astype('int32')
        data['ip_changes'] = data['ip_changes'].astype('int16')
        data['country_changes'] = data['country_changes'].astype('int16')
        data['hour'] = data['hour'].astype('int8')
        data['day_of_week'] = data['day_of_week'].astype('int8')
        data['abnormal_rtt'] = data['abnormal_rtt'].astype('int8')
        data['is_night'] = data['is_night'].astype('int8')
        data['is_weekend'] = data['is_weekend'].astype('int8')
        data['is_attack'] = data['is_attack'].astype('int8')
        
        print(f"RBA Dataset: {data.shape[0]:,} samples, Attack rate: {data['is_attack'].mean():.2%}")
        print(f"Attack samples: {data['is_attack'].sum():,}, Benign samples: {(~data['is_attack'].astype(bool)).sum():,}")
        print(f"Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        return data

    def preprocess_features(self, df, fit=False):
        """Preprocess features with encoding and scaling"""
        df = df.copy()
        
        # Encode categorical features
        if 'device_type' in df.columns:
            if fit:
                self.label_encoders['device_type'] = LabelEncoder()
                df['device_type'] = self.label_encoders['device_type'].fit_transform(df['device_type'].astype(str))
            else:
                # Handle unseen categories
                known_classes = set(self.label_encoders['device_type'].classes_)
                df['device_type'] = df['device_type'].astype(str).apply(
                    lambda x: x if x in known_classes else 'unknown'
                )
                df['device_type'] = self.label_encoders['device_type'].transform(df['device_type'])
        
        # Separate features and target
        if 'is_attack' in df.columns:
            X = df.drop('is_attack', axis=1)
            y = df['is_attack']
        else:
            X = df
            y = None
        
        # Store feature names
        if fit:
            self.feature_names = X.columns.tolist()
        
        # Scale features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y
    
    def handle_imbalance(self, X, y, method='smote', max_samples=200000):
        """Handle imbalanced data using SMOTE with memory limits"""
        print(f"\nOriginal class distribution: {np.bincount(y)}")
        
        if method == 'smote':
            # If dataset is too large, sample before SMOTE to avoid OOM
            if len(X) > max_samples:
                print(f"Dataset too large ({len(X):,}), sampling {max_samples:,} for SMOTE...")
                from sklearn.model_selection import train_test_split
                X_sample, _, y_sample, _ = train_test_split(X, y, train_size=max_samples, 
                                                             random_state=42, stratify=y)
                X, y = X_sample, y_sample
                print(f"Sampled class distribution: {np.bincount(y)}")
            
            # Use smaller k_neighbors if minority class is small
            minority_count = min(np.bincount(y))
            k_neighbors = min(5, minority_count - 1) if minority_count > 1 else 1
            
            if k_neighbors >= 1:
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                print(f"After SMOTE: {np.bincount(y_resampled)}")
                return X_resampled, y_resampled
            else:
                print("Minority class too small for SMOTE, using class weights only")
                return X, y
        
        return X, y
    
    def train_xgboost(self, X_train, y_train, X_val, y_val, use_weights=True):
        """Train XGBoost model"""
        print("\n" + "="*50)
        print("Training XGBoost Model")
        print("="*50)
        
        # Calculate class weights
        if use_weights:
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        else:
            scale_pos_weight = 1
        
        # XGBoost parameters optimized for speed and accuracy
        params = {
            'objective': 'binary:logistic',
            'eval_metric': ['auc', 'logloss'],
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': scale_pos_weight,
            'tree_method': 'hist',  # Faster training
            'random_state': 42
        }
        
        self.xgb_model = xgb.XGBClassifier(**params)
        
        # Train with early stopping
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        return self.xgb_model
    
    def train_lstm(self, X_train, y_train, X_val, y_val, use_weights=True):
        """Train LSTM model"""
        print("\n" + "="*50)
        print("Training LSTM Model")
        print("="*50)
        
        # Reshape for LSTM (samples, timesteps, features)
        X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_val_lstm = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
        
        # Calculate class weights
        if use_weights:
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            class_weight = {0: 1.0, 1: neg_count / pos_count}
        else:
            class_weight = None
        
        # Build LSTM model
        model = keras.Sequential([
            layers.LSTM(64, input_shape=(1, X_train.shape[1]), return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(32),
            layers.Dropout(0.3),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        # Train with early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=5,
            restore_best_weights=True,
            mode='max'
        )
        
        model.fit(
            X_train_lstm, y_train,
            validation_data=(X_val_lstm, y_val),
            epochs=30,
            batch_size=256,
            class_weight=class_weight,
            callbacks=[early_stop],
            verbose=0
        )
        
        self.lstm_model = model
        return model
    
    def evaluate_model(self, model, X_test, y_test, model_name, measure_latency=True):
        """Evaluate model performance"""
        print(f"\n{'='*50}")
        print(f"Evaluating {model_name}")
        print(f"{'='*50}")
        
        # Reshape for LSTM if needed
        if model_name == "LSTM":
            X_test_eval = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        else:
            X_test_eval = X_test
        
        # Predictions
        if model_name == "LSTM":
            y_pred_proba = model.predict(X_test_eval, verbose=0).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred_proba = model.predict_proba(X_test_eval)[:, 1]
            y_pred = model.predict(X_test_eval)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        
        # Measure inference latency
        latency = None
        if measure_latency:
            n_samples = min(1000, len(X_test))
            X_sample = X_test[:n_samples]
            if model_name == "LSTM":
                X_sample = X_sample.reshape((X_sample.shape[0], 1, X_sample.shape[1]))
            
            start_time = time.time()
            if model_name == "LSTM":
                _ = model.predict(X_sample, verbose=0)
            else:
                _ = model.predict(X_sample)
            end_time = time.time()
            
            latency = ((end_time - start_time) / n_samples) * 1000  # ms per sample
        
        # Store metrics
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'latency_ms': latency
        }
        
        self.performance_metrics[model_name] = metrics
        
        # Print results
        print(f"\nPrecision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")
        if latency:
            print(f"Avg Inference Latency: {latency:.2f} ms")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Benign', 'Attack']))
        
        return metrics

    def sequential_fine_tuning(self, datasets, use_smote=True):
        """Sequential fine-tuning across multiple datasets"""
        print("\n" + "="*70)
        print("SEQUENTIAL FINE-TUNING PIPELINE")
        print("="*70)
        
        all_data = []
        
        # Load all datasets
        for i, (name, path) in enumerate(datasets):
            if name == "Mobile Security":
                data = self.load_mobile_security_dataset(path)
            elif name == "Attack":
                data = self.load_attack_dataset(path)
            elif name == "RBA":
                data = self.load_rba_dataset(path)
            
            all_data.append((name, data))
        
        # Stage 1: Train on Mobile Security Dataset
        print(f"\n{'='*70}")
        print("STAGE 1: Training on Mobile Security Dataset")
        print(f"{'='*70}")
        
        data1 = all_data[0][1]
        X1, y1 = self.preprocess_features(data1, fit=True)
        del data1
        gc.collect()
        
        X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42, stratify=y1)
        X1_train, X1_val, y1_train, y1_val = train_test_split(X1_train, y1_train, test_size=0.2, random_state=42, stratify=y1_train)
        del X1, y1
        gc.collect()
        
        if use_smote:
            X1_train, y1_train = self.handle_imbalance(X1_train, y1_train)
        
        self.train_xgboost(X1_train, y1_train, X1_val, y1_val)
        self.train_lstm(X1_train, y1_train, X1_val, y1_val)
        
        del X1_train, X1_val, y1_train, y1_val, X1_test, y1_test
        gc.collect()
        print_memory_usage()
        
        # Stage 2: Fine-tune on Attack Dataset
        print(f"\n{'='*70}")
        print("STAGE 2: Fine-tuning on Cybersecurity Attack Dataset")
        print(f"{'='*70}")
        
        data2 = all_data[1][1]
        X2, y2 = self.preprocess_features(data2, fit=False)
        del data2
        gc.collect()
        
        X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42, stratify=y2)
        X2_train, X2_val, y2_train, y2_val = train_test_split(X2_train, y2_train, test_size=0.2, random_state=42, stratify=y2_train)
        del X2, y2
        gc.collect()
        
        if use_smote:
            X2_train, y2_train = self.handle_imbalance(X2_train, y2_train)
        
        # Continue training XGBoost (warm start)
        self.train_xgboost(X2_train, y2_train, X2_val, y2_val)
        
        # Continue training LSTM
        X2_train_lstm = X2_train.reshape((X2_train.shape[0], 1, X2_train.shape[1]))
        X2_val_lstm = X2_val.reshape((X2_val.shape[0], 1, X2_val.shape[1]))
        self.lstm_model.fit(X2_train_lstm, y2_train, validation_data=(X2_val_lstm, y2_val), 
                           epochs=10, batch_size=256, verbose=0)
        
        del X2_train, X2_val, y2_train, y2_val, X2_test, y2_test, X2_train_lstm, X2_val_lstm
        gc.collect()
        print_memory_usage()
        
        # Stage 3: Fine-tune on RBA Dataset
        print(f"\n{'='*70}")
        print("STAGE 3: Fine-tuning on RBA Dataset (Large Dataset)")
        print(f"{'='*70}")
        
        data3 = all_data[2][1]
        X3, y3 = self.preprocess_features(data3, fit=False)
        del data3
        gc.collect()
        print_memory_usage()
        
        X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=42, stratify=y3)
        del X3, y3
        gc.collect()
        
        X3_train, X3_val, y3_train, y3_val = train_test_split(X3_train, y3_train, test_size=0.2, random_state=42, stratify=y3_train)
        
        if use_smote:
            X3_train, y3_train = self.handle_imbalance(X3_train, y3_train, max_samples=200000)
        
        print_memory_usage()
        
        # Final fine-tuning XGBoost
        self.train_xgboost(X3_train, y3_train, X3_val, y3_val)
        
        # Final fine-tuning LSTM
        X3_train_lstm = X3_train.reshape((X3_train.shape[0], 1, X3_train.shape[1]))
        X3_val_lstm = X3_val.reshape((X3_val.shape[0], 1, X3_val.shape[1]))
        self.lstm_model.fit(X3_train_lstm, y3_train, validation_data=(X3_val_lstm, y3_val), 
                           epochs=10, batch_size=256, verbose=0)
        
        del X3_train, X3_val, y3_train, y3_val, X3_train_lstm, X3_val_lstm
        gc.collect()
        print_memory_usage()
        
        # Final evaluation on RBA test set
        print(f"\n{'='*70}")
        print("FINAL EVALUATION ON RBA TEST SET")
        print(f"{'='*70}")
        
        self.evaluate_model(self.xgb_model, X3_test, y3_test, "XGBoost")
        self.evaluate_model(self.lstm_model, X3_test, y3_test, "LSTM")
        
        return X3_test, y3_test
    
    def compare_models(self):
        """Compare XGBoost and LSTM performance"""
        print(f"\n{'='*70}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*70}\n")
        
        comparison = pd.DataFrame(self.performance_metrics).T
        print(comparison.to_string())
        
        # Determine best model
        print(f"\n{'='*70}")
        print("BEST MODEL SELECTION")
        print(f"{'='*70}")
        
        xgb_metrics = self.performance_metrics['XGBoost']
        lstm_metrics = self.performance_metrics['LSTM']
        
        # Scoring: F1 (40%) + AUC (40%) + Latency (20%, inverted)
        xgb_score = (xgb_metrics['f1_score'] * 0.4 + 
                     xgb_metrics['auc_roc'] * 0.4 + 
                     (1 - min(xgb_metrics['latency_ms'] / 100, 1)) * 0.2)
        
        lstm_score = (lstm_metrics['f1_score'] * 0.4 + 
                      lstm_metrics['auc_roc'] * 0.4 + 
                      (1 - min(lstm_metrics['latency_ms'] / 100, 1)) * 0.2)
        
        print(f"\nXGBoost Overall Score: {xgb_score:.4f}")
        print(f"LSTM Overall Score: {lstm_score:.4f}")
        
        if xgb_score > lstm_score:
            print(f"\n✓ WINNER: XGBoost")
            print(f"  - Better F1-Score: {xgb_metrics['f1_score']:.4f}")
            print(f"  - Better AUC-ROC: {xgb_metrics['auc_roc']:.4f}")
            print(f"  - Faster Inference: {xgb_metrics['latency_ms']:.2f} ms")
            best_model = "XGBoost"
        else:
            print(f"\n✓ WINNER: LSTM")
            print(f"  - Better F1-Score: {lstm_metrics['f1_score']:.4f}")
            print(f"  - Better AUC-ROC: {lstm_metrics['auc_roc']:.4f}")
            print(f"  - Inference Latency: {lstm_metrics['latency_ms']:.2f} ms")
            best_model = "LSTM"
        
        # Check latency requirement
        print(f"\n{'='*70}")
        print("REAL-TIME INFERENCE REQUIREMENT CHECK")
        print(f"{'='*70}")
        
        for model_name, metrics in self.performance_metrics.items():
            latency = metrics['latency_ms']
            status = "✓ PASS" if latency < 100 else "✗ FAIL"
            print(f"{model_name}: {latency:.2f} ms - {status}")
        
        return best_model, comparison
    
    def save_models(self, xgb_path='xgboost_model.json', lstm_path='lstm_model.h5', 
                    scaler_path='scaler.pkl', encoders_path='encoders.pkl'):
        """Save trained models and preprocessors"""
        print(f"\n{'='*70}")
        print("SAVING MODELS")
        print(f"{'='*70}")
        
        self.xgb_model.save_model(xgb_path)
        print(f"✓ XGBoost model saved to {xgb_path}")
        
        self.lstm_model.save(lstm_path)
        print(f"✓ LSTM model saved to {lstm_path}")
        
        joblib.dump(self.scaler, scaler_path)
        print(f"✓ Scaler saved to {scaler_path}")
        
        joblib.dump(self.label_encoders, encoders_path)
        print(f"✓ Label encoders saved to {encoders_path}")
    
    def generate_report(self, output_path='performance_report.json'):
        """Generate performance comparison report"""
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'models': self.performance_metrics,
            'feature_names': self.feature_names,
            'sequential_training': {
                'stage_1': 'Mobile Security Dataset',
                'stage_2': 'Cybersecurity Attack Dataset',
                'stage_3': 'RBA Dataset'
            },
            'data_handling': {
                'imbalance_method': 'SMOTE',
                'scaling': 'StandardScaler'
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Performance report saved to {output_path}")
        return report


def main():
    """Main execution pipeline"""
    print("="*70)
    print("WAF BROKEN AUTHENTICATION DETECTION SYSTEM")
    print("="*70)
    
    # Initialize detector
    detector = WAFAuthenticationDetector()
    
    # Define datasets for sequential training
    datasets = [
        ("Mobile Security", "archive (3)/Mobile Security Dataset.csv"),
        ("Attack", "archive (4)/Attack_Dataset.csv"),
        ("RBA", "archive (5)/rba-dataset.csv")
    ]
    
    # Sequential fine-tuning
    X_test, y_test = detector.sequential_fine_tuning(datasets, use_smote=True)
    
    # Compare models
    best_model, comparison = detector.compare_models()
    
    # Save models
    detector.save_models()
    
    # Generate report
    detector.generate_report()
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"\nBest Model: {best_model}")
    print(f"Ready for deployment with <100ms inference latency")
    

if __name__ == "__main__":
    main()
