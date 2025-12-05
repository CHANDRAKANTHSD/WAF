"""
XGBoost-based WAF for LDAP Injection Detection
Trains on multiple datasets consecutively with feature extraction and SMOTE
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_auc_score, roc_curve,
                             classification_report)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

warnings.filterwarnings('ignore')

class XGBoostWAF:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.training_history = []
        
    def extract_features(self, df, dataset_name):
        """Extract and engineer features from different datasets"""
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name} Dataset")
        print(f"{'='*60}")
        print(f"Original shape: {df.shape}")
        
        # Handle different dataset formats
        if dataset_name == "CICDDoS2019":
            # Already has 80+ network features
            feature_cols = [col for col in df.columns if col != 'Label']
            X = df[feature_cols].copy()
            y = df['Label'].copy()
            
            # Encode labels
            y = (y != 'NetBIOS').astype(int)  # 1 for attack, 0 for benign
            
        elif dataset_name == "LSNM2024":
            # Network packet features
            feature_cols = [col for col in df.columns if col not in ['No.', 'Time', 'label', 
                           'Source', 'Destination', 'Info', 'Frame Time', 'Frame Protocols',
                           'Ethernet Source', 'Ethernet Destination', 'IP Source', 'IP Destination',
                           'HTTP Request URI', 'HTTP Full URI', 'HTTP User-Agent', 'HTTP Cookie',
                           'HTTP Host', 'HTTP Referer', 'HTTP Location', 'HTTP Authorization',
                           'DNS Query Name']]
            
            X = df[feature_cols].copy()
            y = df['label'].copy()
            
            # Encode labels
            y = (y != 'normal').astype(int)
            
        elif dataset_name == "CSIC":
            # HTTP request features - need to engineer features
            X = pd.DataFrame()
            
            # Length-based features
            X['url_length'] = df['URL'].fillna('').str.len()
            X['content_length'] = df['lenght'].fillna(0)
            X['method_encoded'] = LabelEncoder().fit_transform(df['Method'].fillna('GET'))
            
            # Character frequency features
            X['special_char_count'] = df['URL'].fillna('').str.count(r'[^a-zA-Z0-9\s]')
            X['digit_count'] = df['URL'].fillna('').str.count(r'\d')
            X['uppercase_count'] = df['URL'].fillna('').str.count(r'[A-Z]')
            X['lowercase_count'] = df['URL'].fillna('').str.count(r'[a-z]')
            
            # SQL injection indicators
            X['sql_keywords'] = df['URL'].fillna('').str.lower().str.count(
                r'(select|union|insert|update|delete|drop|create|alter|exec|script)')
            X['has_quotes'] = df['URL'].fillna('').str.contains(r"['\"]").astype(int)
            X['has_comment'] = df['URL'].fillna('').str.contains(r'(--|#|/\*)').astype(int)
            X['has_semicolon'] = df['URL'].fillna('').str.contains(';').astype(int)
            
            # LDAP injection indicators
            X['ldap_chars'] = df['URL'].fillna('').str.count(r'[\(\)\*\|&]')
            X['has_wildcard'] = df['URL'].fillna('').str.contains(r'\*').astype(int)
            
            # Entropy
            def calculate_entropy(s):
                if not s:
                    return 0
                prob = [float(s.count(c)) / len(s) for c in dict.fromkeys(list(s))]
                return -sum([p * np.log2(p) for p in prob])
            
            X['url_entropy'] = df['URL'].fillna('').apply(calculate_entropy)
            
            # User-Agent features
            X['user_agent_length'] = df['User-Agent'].fillna('').str.len()
            X['has_user_agent'] = (df['User-Agent'].notna()).astype(int)
            
            y = df['classification'].copy()
            
        # Handle missing values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # Select numeric columns only
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        print(f"Features extracted: {X.shape[1]}")
        print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")
        
        return X, y
    
    def train_on_dataset(self, X, y, dataset_name, test_size=0.15, val_size=0.15):
        """Train model on a single dataset"""
        print(f"\n{'='*60}")
        print(f"Training on {dataset_name}")
        print(f"{'='*60}")
        
        # Split data: 70% train, 15% val, 15% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
        )
        
        print(f"Train set: {X_train.shape}, Val set: {X_val.shape}, Test set: {X_test.shape}")
        
        # Handle class imbalance with SMOTE
        print("\nApplying SMOTE for class balancing...")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE: {X_train_balanced.shape}")
        print(f"Class distribution: {pd.Series(y_train_balanced).value_counts().to_dict()}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_balanced)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost model
        if self.model is None:
            print("\nTraining NEW XGBoost classifier...")
            self.model = xgb.XGBClassifier(
                max_depth=6,
                learning_rate=0.1,
                n_estimators=200,
                objective='binary:logistic',
                eval_metric='auc',
                random_state=42,
                n_jobs=-1
            )
            # Initial training
            self.model.fit(
                X_train_scaled, y_train_balanced,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False
            )
        else:
            print("\nFINE-TUNING existing XGBoost classifier on new dataset...")
            # Continue training with xgb_model parameter for incremental learning
            self.model.fit(
                X_train_scaled, y_train_balanced,
                eval_set=[(X_val_scaled, y_val)],
                xgb_model=self.model.get_booster(),
                verbose=False
            )
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Evaluate
        results = self.evaluate(X_test_scaled, y_test, dataset_name)
        
        # Store training history
        self.training_history.append({
            'dataset': dataset_name,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'results': results
        })
        
        return results
    
    def evaluate(self, X_test, y_test, dataset_name):
        """Evaluate model performance"""
        print(f"\n{'='*60}")
        print(f"Evaluation Results for {dataset_name}")
        print(f"{'='*60}")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # False Positive Rate
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # ROC-AUC
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except:
            roc_auc = 0.0
        
        print(f"\nAccuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print(f"FPR:       {fpr:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Benign', 'Attack']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Benign', 'Attack'],
                   yticklabels=['Benign', 'Attack'])
        plt.title(f'Confusion Matrix - {dataset_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{dataset_name}_xgboost.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved: confusion_matrix_{dataset_name}_xgboost.png")
        
        # ROC Curve
        if roc_auc > 0:
            fpr_curve, tpr_curve, _ = roc_curve(y_test, y_pred_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr_curve, tpr_curve, label=f'ROC Curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {dataset_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'roc_curve_{dataset_name}_xgboost.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"ROC curve saved: roc_curve_{dataset_name}_xgboost.png")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'fpr': fpr,
            'confusion_matrix': cm.tolist()
        }
    
    def plot_feature_importance(self, top_n=20):
        """Plot feature importance"""
        if self.model is None or self.feature_names is None:
            print("Model not trained yet!")
            return
        
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        # Limit to available features
        actual_top_n = min(top_n, len(importance))
        indices = indices[:actual_top_n]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(actual_top_n), importance[indices])
        plt.yticks(range(actual_top_n), [self.feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {actual_top_n} Feature Importance - XGBoost')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance_xgboost.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nFeature importance plot saved: feature_importance_xgboost.png")
    
    def predict_realtime(self, query_features):
        """Real-time prediction for incoming LDAP queries"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Ensure features match training
        query_df = pd.DataFrame([query_features])
        
        # Handle missing features
        for feat in self.feature_names:
            if feat not in query_df.columns:
                query_df[feat] = 0
        
        query_df = query_df[self.feature_names]
        
        # Scale and predict
        query_scaled = self.scaler.transform(query_df)
        prediction = self.model.predict(query_scaled)[0]
        probability = self.model.predict_proba(query_scaled)[0]
        
        return {
            'is_attack': bool(prediction),
            'attack_probability': float(probability[1]),
            'benign_probability': float(probability[0])
        }
    
    def save_model(self, filename='xgboost_waf_model.pkl'):
        """Save trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'training_history': self.training_history
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nModel saved: {filename}")
        print(f"File size: {os.path.getsize(filename) / 1024:.2f} KB")
    
    @staticmethod
    def load_model(filename='xgboost_waf_model.pkl'):
        """Load trained model"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        waf = XGBoostWAF()
        waf.model = model_data['model']
        waf.scaler = model_data['scaler']
        waf.feature_names = model_data['feature_names']
        waf.training_history = model_data['training_history']
        
        return waf


def main():
    print("="*60)
    print("XGBoost WAF for LDAP Injection Detection")
    print("="*60)
    
    waf = XGBoostWAF()
    all_results = {}
    
    # Dataset 1: CICDDoS2019 LDAP
    try:
        print("\n\n" + "="*60)
        print("DATASET 1: CICDDoS2019 LDAP")
        print("="*60)
        
        df_train = pd.read_parquet('cicddos_2019/LDAP-training.parquet')
        df_test = pd.read_parquet('cicddos_2019/LDAP-testing.parquet')
        df_cicddos = pd.concat([df_train, df_test], ignore_index=True)
        
        X, y = waf.extract_features(df_cicddos, "CICDDoS2019")
        results = waf.train_on_dataset(X, y, "CICDDoS2019")
        all_results['CICDDoS2019'] = results
        
    except Exception as e:
        print(f"Error processing CICDDoS2019: {e}")
    
    # Dataset 2: LSNM2024
    try:
        print("\n\n" + "="*60)
        print("DATASET 2: LSNM2024")
        print("="*60)
        
        # Load all LSNM2024 files with sample to avoid memory issues
        df_benign = pd.read_csv('LSNM2024 Dataset/Benign/normal_data.csv', nrows=5000)
        df_fuzzing = pd.read_csv('LSNM2024 Dataset/Malicious/Fuzzing/Fuzzing.csv', nrows=5000)
        df_sql1 = pd.read_csv('LSNM2024 Dataset/Malicious/SQL injection/SQL Injection.csv', nrows=5000)
        df_sql2 = pd.read_csv('LSNM2024 Dataset/Malicious/SQL injection/SQL-Injection2.csv', nrows=5000)
        df_sql3 = pd.read_csv('LSNM2024 Dataset/Malicious/SQL injection/SQLinjection-UPDATED.csv', nrows=5000)
        
        df_lsnm = pd.concat([df_benign, df_fuzzing, df_sql1, df_sql2, df_sql3], ignore_index=True)
        
        X, y = waf.extract_features(df_lsnm, "LSNM2024")
        results = waf.train_on_dataset(X, y, "LSNM2024")
        all_results['LSNM2024'] = results
        
    except Exception as e:
        print(f"Error processing LSNM2024: {e}")
    
    # Dataset 3: CSIC
    try:
        print("\n\n" + "="*60)
        print("DATASET 3: CSIC Database")
        print("="*60)
        
        df_csic = pd.read_csv('csic_database.csv')
        
        X, y = waf.extract_features(df_csic, "CSIC")
        results = waf.train_on_dataset(X, y, "CSIC")
        all_results['CSIC'] = results
        
    except Exception as e:
        print(f"Error processing CSIC: {e}")
    
    # Plot feature importance
    waf.plot_feature_importance()
    
    # Save model
    waf.save_model('xgboost_waf_model.pkl')
    
    # Print summary
    print("\n\n" + "="*60)
    print("TRAINING SUMMARY - XGBoost WAF")
    print("="*60)
    
    for dataset, results in all_results.items():
        print(f"\n{dataset}:")
        print(f"  Accuracy:  {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1-Score:  {results['f1_score']:.4f}")
        print(f"  ROC-AUC:   {results['roc_auc']:.4f}")
        print(f"  FPR:       {results['fpr']:.4f}")
    
    # Demo real-time prediction
    print("\n\n" + "="*60)
    print("REAL-TIME PREDICTION DEMO")
    print("="*60)
    
    # Example query features
    sample_query = {feat: 0 for feat in waf.feature_names}
    sample_query[waf.feature_names[0]] = 100  # Set some values
    
    prediction = waf.predict_realtime(sample_query)
    print(f"\nSample Query Prediction:")
    print(f"  Is Attack: {prediction['is_attack']}")
    print(f"  Attack Probability: {prediction['attack_probability']:.4f}")
    print(f"  Benign Probability: {prediction['benign_probability']:.4f}")
    
    print("\n" + "="*60)
    print("XGBoost WAF Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
