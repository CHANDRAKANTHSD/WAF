"""
XGBoost-based WAF for LDAP Injection Detection - UNIFIED MODEL
Combines all datasets with unified feature set and trains a single model
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

class UnifiedXGBoostWAF:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.training_history = []
        
    def extract_unified_features(self, df, dataset_name):
        """Extract unified features that work across all datasets"""
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name} Dataset")
        print(f"{'='*60}")
        print(f"Original shape: {df.shape}")
        
        # Create unified feature set
        X = pd.DataFrame()
        
        if dataset_name == "CICDDoS2019":
            # Network flow features
            if 'Flow Duration' in df.columns:
                X['flow_duration'] = df['Flow Duration']
            if 'Total Fwd Packets' in df.columns:
                X['total_fwd_packets'] = df['Total Fwd Packets']
            if 'Total Backward Packets' in df.columns:
                X['total_bwd_packets'] = df['Total Backward Packets']
            if 'Fwd Packets Length Total' in df.columns:
                X['fwd_packet_length'] = df['Fwd Packets Length Total']
            if 'Bwd Packets Length Total' in df.columns:
                X['bwd_packet_length'] = df['Bwd Packets Length Total']
            if 'Flow Bytes/s' in df.columns:
                X['flow_bytes_per_sec'] = df['Flow Bytes/s'].replace([np.inf, -np.inf], 0)
            if 'Flow Packets/s' in df.columns:
                X['flow_packets_per_sec'] = df['Flow Packets/s'].replace([np.inf, -np.inf], 0)
            
            # Statistical features
            if 'Fwd Packet Length Mean' in df.columns:
                X['fwd_pkt_len_mean'] = df['Fwd Packet Length Mean']
            if 'Bwd Packet Length Mean' in df.columns:
                X['bwd_pkt_len_mean'] = df['Bwd Packet Length Mean']
            if 'Fwd Packet Length Std' in df.columns:
                X['fwd_pkt_len_std'] = df['Fwd Packet Length Std']
            
            y = (df['Label'] != 'NetBIOS').astype(int)
            
        elif dataset_name == "LSNM2024":
            # Packet-level features
            if 'Length' in df.columns:
                X['packet_length'] = df['Length']
            if 'frame length' in df.columns:
                X['frame_length'] = df['frame length']
            if 'IP Length' in df.columns:
                X['ip_length'] = df['IP Length']
            if 'TCP Length' in df.columns:
                X['tcp_length'] = df['TCP Length'].fillna(0)
            if 'UDP Length' in df.columns:
                X['udp_length'] = df['UDP Length'].fillna(0)
            
            # Protocol features
            if 'TCP SYN Flag' in df.columns:
                X['tcp_syn'] = df['TCP SYN Flag'].fillna(0)
            if 'TCP ACK Flag' in df.columns:
                X['tcp_ack'] = df['TCP ACK Flag'].fillna(0)
            if 'TCP FIN Flag' in df.columns:
                X['tcp_fin'] = df['TCP FIN Flag'].fillna(0)
            if 'TCP RST Flag' in df.columns:
                X['tcp_rst'] = df['TCP RST Flag'].fillna(0)
            if 'TCP Window Size' in df.columns:
                X['tcp_window'] = df['TCP Window Size'].fillna(0)
            
            y = (df['label'] != 'normal').astype(int)
            
        elif dataset_name == "CSIC":
            # HTTP request features
            X['url_length'] = df['URL'].fillna('').str.len()
            X['content_length'] = df['lenght'].fillna(0)
            X['method_encoded'] = LabelEncoder().fit_transform(df['Method'].fillna('GET'))
            
            # Character analysis
            X['special_char_count'] = df['URL'].fillna('').str.count(r'[^a-zA-Z0-9\s]')
            X['digit_count'] = df['URL'].fillna('').str.count(r'\d')
            X['uppercase_count'] = df['URL'].fillna('').str.count(r'[A-Z]')
            X['lowercase_count'] = df['URL'].fillna('').str.count(r'[a-z]')
            
            # Attack indicators
            X['sql_keywords'] = df['URL'].fillna('').str.lower().str.count(
                r'(select|union|insert|update|delete|drop|create|alter|exec|script)')
            X['has_quotes'] = df['URL'].fillna('').str.contains(r"['\"]").astype(int)
            X['has_comment'] = df['URL'].fillna('').str.contains(r'(--|#|/\*)').astype(int)
            
            y = df['classification'].copy()
        
        # Add dataset identifier as feature
        X['dataset_id'] = {'CICDDoS2019': 0, 'LSNM2024': 1, 'CSIC': 2}[dataset_name]
        
        # Fill missing columns with zeros to create unified feature space
        all_possible_features = [
            'flow_duration', 'total_fwd_packets', 'total_bwd_packets',
            'fwd_packet_length', 'bwd_packet_length', 'flow_bytes_per_sec',
            'flow_packets_per_sec', 'fwd_pkt_len_mean', 'bwd_pkt_len_mean',
            'fwd_pkt_len_std', 'packet_length', 'frame_length', 'ip_length',
            'tcp_length', 'udp_length', 'tcp_syn', 'tcp_ack', 'tcp_fin',
            'tcp_rst', 'tcp_window', 'url_length', 'content_length',
            'method_encoded', 'special_char_count', 'digit_count',
            'uppercase_count', 'lowercase_count', 'sql_keywords',
            'has_quotes', 'has_comment', 'dataset_id'
        ]
        
        for feat in all_possible_features:
            if feat not in X.columns:
                X[feat] = 0
        
        # Reorder columns
        X = X[all_possible_features]
        
        # Handle missing values and ensure numeric types
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Convert all columns to numeric, coercing errors
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        X = X.fillna(0)
        
        print(f"Unified features: {X.shape[1]}")
        print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")
        
        return X, y
    
    def train_unified_model(self, all_X, all_y):
        """Train a single model on all combined data"""
        print(f"\n{'='*60}")
        print(f"Training Unified Model on Combined Data")
        print(f"{'='*60}")
        print(f"Total samples: {len(all_y)}")
        
        # Split data: 70% train, 15% val, 15% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            all_X, all_y, test_size=0.15, random_state=42, stratify=all_y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
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
        print("\nTraining Unified XGBoost classifier...")
        self.model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            objective='binary:logistic',
            eval_metric='auc',
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(
            X_train_scaled, y_train_balanced,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False
        )
        
        self.feature_names = all_X.columns.tolist()
        
        # Evaluate
        return self.evaluate(X_test_scaled, y_test, X_test, all_X)
    
    def evaluate(self, X_test_scaled, y_test, X_test_original, all_X):
        """Evaluate model performance overall and per dataset"""
        print(f"\n{'='*60}")
        print(f"Unified Model Evaluation")
        print(f"{'='*60}")
        
        # Overall predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Overall metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except:
            roc_auc = 0.0
        
        print(f"\nOVERALL PERFORMANCE:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print(f"FPR:       {fpr:.4f}")
        
        # Per-dataset performance
        print(f"\nPER-DATASET PERFORMANCE:")
        dataset_ids = X_test_original['dataset_id'].values
        
        for ds_id, ds_name in [(0, 'CICDDoS2019'), (1, 'LSNM2024'), (2, 'CSIC')]:
            mask = dataset_ids == ds_id
            if mask.sum() > 0:
                ds_acc = accuracy_score(y_test[mask], y_pred[mask])
                ds_prec = precision_score(y_test[mask], y_pred[mask], zero_division=0)
                ds_rec = recall_score(y_test[mask], y_pred[mask], zero_division=0)
                ds_f1 = f1_score(y_test[mask], y_pred[mask], zero_division=0)
                
                print(f"\n{ds_name}:")
                print(f"  Samples: {mask.sum()}")
                print(f"  Accuracy:  {ds_acc:.4f}")
                print(f"  Precision: {ds_prec:.4f}")
                print(f"  Recall:    {ds_rec:.4f}")
                print(f"  F1-Score:  {ds_f1:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Benign', 'Attack'],
                   yticklabels=['Benign', 'Attack'])
        plt.title('Confusion Matrix - Unified Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix_unified_xgboost.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nConfusion matrix saved: confusion_matrix_unified_xgboost.png")
        
        # ROC Curve
        if roc_auc > 0:
            fpr_curve, tpr_curve, _ = roc_curve(y_test, y_pred_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr_curve, tpr_curve, label=f'ROC Curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve - Unified Model')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('roc_curve_unified_xgboost.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"ROC curve saved: roc_curve_unified_xgboost.png")
        
        # Feature importance
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1][:20]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(20), importance[indices])
        plt.yticks(range(20), [self.feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Feature Importance - Unified Model')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance_unified_xgboost.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature importance saved: feature_importance_unified_xgboost.png")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'fpr': fpr
        }
    
    def save_model(self, filename='xgboost_waf_unified.pkl'):
        """Save trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'training_history': self.training_history
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nUnified model saved: {filename}")
        print(f"File size: {os.path.getsize(filename) / 1024:.2f} KB")


def main():
    print("="*60)
    print("XGBoost WAF - UNIFIED MODEL (All Datasets Combined)")
    print("="*60)
    
    waf = UnifiedXGBoostWAF()
    all_X_list = []
    all_y_list = []
    
    # Load and process all datasets
    print("\n" + "="*60)
    print("LOADING ALL DATASETS")
    print("="*60)
    
    # Dataset 1: CICDDoS2019
    try:
        df_train = pd.read_parquet('cicddos_2019/LDAP-training.parquet')
        df_test = pd.read_parquet('cicddos_2019/LDAP-testing.parquet')
        df_cicddos = pd.concat([df_train, df_test], ignore_index=True)
        
        X, y = waf.extract_unified_features(df_cicddos, "CICDDoS2019")
        all_X_list.append(X)
        all_y_list.append(y)
    except Exception as e:
        print(f"Error loading CICDDoS2019: {e}")
    
    # Dataset 2: LSNM2024
    try:
        df_benign = pd.read_csv('LSNM2024 Dataset/Benign/normal_data.csv', nrows=5000)
        df_fuzzing = pd.read_csv('LSNM2024 Dataset/Malicious/Fuzzing/Fuzzing.csv', nrows=5000)
        df_sql1 = pd.read_csv('LSNM2024 Dataset/Malicious/SQL injection/SQL Injection.csv', nrows=5000)
        df_sql2 = pd.read_csv('LSNM2024 Dataset/Malicious/SQL injection/SQL-Injection2.csv', nrows=5000)
        df_sql3 = pd.read_csv('LSNM2024 Dataset/Malicious/SQL injection/SQLinjection-UPDATED.csv', nrows=5000)
        
        df_lsnm = pd.concat([df_benign, df_fuzzing, df_sql1, df_sql2, df_sql3], ignore_index=True)
        
        X, y = waf.extract_unified_features(df_lsnm, "LSNM2024")
        all_X_list.append(X)
        all_y_list.append(y)
    except Exception as e:
        print(f"Error loading LSNM2024: {e}")
    
    # Dataset 3: CSIC
    try:
        df_csic = pd.read_csv('csic_database.csv')
        
        X, y = waf.extract_unified_features(df_csic, "CSIC")
        all_X_list.append(X)
        all_y_list.append(y)
    except Exception as e:
        print(f"Error loading CSIC: {e}")
    
    # Combine all datasets
    print("\n" + "="*60)
    print("COMBINING ALL DATASETS")
    print("="*60)
    
    all_X = pd.concat(all_X_list, ignore_index=True)
    all_y = pd.concat(all_y_list, ignore_index=True)
    
    print(f"Combined dataset shape: {all_X.shape}")
    print(f"Total samples: {len(all_y)}")
    print(f"Class distribution: {pd.Series(all_y).value_counts().to_dict()}")
    
    # Train unified model
    results = waf.train_unified_model(all_X, all_y)
    
    # Save model
    waf.save_model('xgboost_waf_unified.pkl')
    
    print("\n" + "="*60)
    print("UNIFIED MODEL TRAINING COMPLETE!")
    print("="*60)
    print("\nThis model has learned from ALL three datasets simultaneously")
    print("and can detect attacks from CICDDoS2019, LSNM2024, and CSIC.")


if __name__ == "__main__":
    main()
