"""
XGBoost-based WAF for LDAP Injection Detection - SEPARATE MODELS VERSION
Trains individual models for each dataset (not fine-tuned)
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

# Import the main XGBoostWAF class
from xgboost_waf_ldap import XGBoostWAF

def train_separate_models():
    """Train separate models for each dataset"""
    print("="*60)
    print("XGBoost WAF - SEPARATE MODELS (No Fine-tuning)")
    print("="*60)
    
    models = {}
    all_results = {}
    
    # Dataset 1: CICDDoS2019 LDAP
    try:
        print("\n\n" + "="*60)
        print("DATASET 1: CICDDoS2019 LDAP")
        print("="*60)
        
        waf_cicddos = XGBoostWAF()
        
        df_train = pd.read_parquet('cicddos_2019/LDAP-training.parquet')
        df_test = pd.read_parquet('cicddos_2019/LDAP-testing.parquet')
        df_cicddos = pd.concat([df_train, df_test], ignore_index=True)
        
        X, y = waf_cicddos.extract_features(df_cicddos, "CICDDoS2019")
        results = waf_cicddos.train_on_dataset(X, y, "CICDDoS2019")
        
        models['CICDDoS2019'] = waf_cicddos
        all_results['CICDDoS2019'] = results
        
        # Save individual model
        waf_cicddos.save_model('xgboost_waf_cicddos_only.pkl')
        
    except Exception as e:
        print(f"Error processing CICDDoS2019: {e}")
    
    # Dataset 2: LSNM2024
    try:
        print("\n\n" + "="*60)
        print("DATASET 2: LSNM2024")
        print("="*60)
        
        waf_lsnm = XGBoostWAF()
        
        df_benign = pd.read_csv('LSNM2024 Dataset/Benign/normal_data.csv', nrows=5000)
        df_fuzzing = pd.read_csv('LSNM2024 Dataset/Malicious/Fuzzing/Fuzzing.csv', nrows=5000)
        df_sql1 = pd.read_csv('LSNM2024 Dataset/Malicious/SQL injection/SQL Injection.csv', nrows=5000)
        df_sql2 = pd.read_csv('LSNM2024 Dataset/Malicious/SQL injection/SQL-Injection2.csv', nrows=5000)
        df_sql3 = pd.read_csv('LSNM2024 Dataset/Malicious/SQL injection/SQLinjection-UPDATED.csv', nrows=5000)
        
        df_lsnm = pd.concat([df_benign, df_fuzzing, df_sql1, df_sql2, df_sql3], ignore_index=True)
        
        X, y = waf_lsnm.extract_features(df_lsnm, "LSNM2024")
        results = waf_lsnm.train_on_dataset(X, y, "LSNM2024")
        
        models['LSNM2024'] = waf_lsnm
        all_results['LSNM2024'] = results
        
        # Save individual model
        waf_lsnm.save_model('xgboost_waf_lsnm_only.pkl')
        
    except Exception as e:
        print(f"Error processing LSNM2024: {e}")
    
    # Dataset 3: CSIC
    try:
        print("\n\n" + "="*60)
        print("DATASET 3: CSIC Database")
        print("="*60)
        
        waf_csic = XGBoostWAF()
        
        df_csic = pd.read_csv('csic_database.csv')
        
        X, y = waf_csic.extract_features(df_csic, "CSIC")
        results = waf_csic.train_on_dataset(X, y, "CSIC")
        
        models['CSIC'] = waf_csic
        all_results['CSIC'] = results
        
        # Save individual model
        waf_csic.save_model('xgboost_waf_csic_only.pkl')
        
    except Exception as e:
        print(f"Error processing CSIC: {e}")
    
    # Print summary
    print("\n\n" + "="*60)
    print("TRAINING SUMMARY - SEPARATE MODELS")
    print("="*60)
    
    for dataset, results in all_results.items():
        print(f"\n{dataset}:")
        print(f"  Accuracy:  {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1-Score:  {results['f1_score']:.4f}")
        print(f"  ROC-AUC:   {results['roc_auc']:.4f}")
        print(f"  FPR:       {results['fpr']:.4f}")
        print(f"  Model saved: xgboost_waf_{dataset.lower()}_only.pkl")
    
    print("\n" + "="*60)
    print("All models trained and saved separately!")
    print("="*60)
    
    return models, all_results


if __name__ == "__main__":
    train_separate_models()
