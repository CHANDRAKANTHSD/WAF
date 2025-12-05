"""
WAF Privilege Escalation Detection - Model Loading and Inference Script
This script demonstrates how to load the trained models and make predictions
"""

import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

print("="*80)
print("Loading Trained Models for WAF Privilege Escalation Detection")
print("="*80)

# ============================================================================
# Load Models
# ============================================================================

print("\n[1/3] Loading models...")

# Load CatBoost model
catboost_model = CatBoostClassifier()
catboost_model.load_model('model/catboost_waf_model.cbm')
print("✓ CatBoost model loaded")

# Load LightGBM model
with open('model/lightgbm_waf_model.pkl', 'rb') as f:
    lightgbm_model = pickle.load(f)
print("✓ LightGBM model loaded")

# Load label encoders for LightGBM
with open('model/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)
print("✓ Label encoders loaded")

# Load feature info
with open('model/feature_info.pkl', 'rb') as f:
    feature_info = pickle.load(f)
print("✓ Feature info loaded")

print(f"\nCategorical features: {feature_info['categorical_features']}")
print(f"Numerical features: {feature_info['numerical_features']}")

# ============================================================================
# Create Sample Data for Prediction
# ============================================================================

print("\n[2/3] Creating sample data for prediction...")

# Sample 1: Privilege escalation attack
sample_1 = {
    'attack_category': 'IAM Misconfiguration',
    'attack_type': 'Privilege Escalation',
    'target_system': 'AWS',
    'mitre_technique': 'T1078 (Valid Accounts)',
    'packet_size': 0.5,
    'inter_arrival_time': 0.3,
    'packet_count_5s': 0.8,
    'mean_packet_size': 0.0,
    'spectral_entropy': 0.7,
    'frequency_band_energy': 0.6
}

# Sample 2: Normal network traffic
sample_2 = {
    'attack_category': 'Network_Security',
    'attack_type': 'Network_Attack',
    'target_system': 'Embedded_System',
    'mitre_technique': 'Network_Intrusion',
    'packet_size': 0.2,
    'inter_arrival_time': 0.1,
    'packet_count_5s': 0.3,
    'mean_packet_size': 0.0,
    'spectral_entropy': 0.2,
    'frequency_band_energy': 0.1
}

# Sample 3: Cloud vulnerability
sample_3 = {
    'attack_category': 'Access Control',
    'attack_type': 'IAM Misconfiguration',
    'target_system': 'Azure',
    'mitre_technique': 'Cloud_Access Control',
    'packet_size': 0.0,
    'inter_arrival_time': 0.0,
    'packet_count_5s': 0.0,
    'mean_packet_size': 0.0,
    'spectral_entropy': 0.0,
    'frequency_band_energy': 0.0
}

samples = pd.DataFrame([sample_1, sample_2, sample_3])
print(f"✓ Created {len(samples)} sample records")

# ============================================================================
# Make Predictions
# ============================================================================

print("\n[3/3] Making predictions...")

# Prepare data for LightGBM (encode categorical features)
samples_lgb = samples.copy()
for col in feature_info['categorical_features']:
    if col in label_encoders:
        # Handle unseen categories
        samples_lgb[col] = samples_lgb[col].apply(
            lambda x: label_encoders[col].transform([str(x)])[0] 
            if str(x) in label_encoders[col].classes_ 
            else 0
        )

print("\n" + "="*80)
print("PREDICTION RESULTS")
print("="*80)

for idx, row in samples.iterrows():
    print(f"\n--- Sample {idx + 1} ---")
    print(f"Attack Category: {row['attack_category']}")
    print(f"Attack Type: {row['attack_type']}")
    print(f"Target System: {row['target_system']}")
    print(f"MITRE Technique: {row['mitre_technique']}")
    
    # CatBoost prediction
    cb_pred = catboost_model.predict(samples.iloc[[idx]])
    cb_proba = catboost_model.predict_proba(samples.iloc[[idx]])[0][1]
    
    # LightGBM prediction
    lgb_pred = lightgbm_model.predict(samples_lgb.iloc[[idx]])
    lgb_proba = lightgbm_model.predict_proba(samples_lgb.iloc[[idx]])[0][1]
    
    print(f"\nCatBoost:")
    print(f"  Prediction: {'PRIVILEGE ESCALATION' if cb_pred[0] == 1 else 'NORMAL'}")
    print(f"  Confidence: {cb_proba:.4f} ({cb_proba*100:.2f}%)")
    
    print(f"\nLightGBM:")
    print(f"  Prediction: {'PRIVILEGE ESCALATION' if lgb_pred[0] == 1 else 'NORMAL'}")
    print(f"  Confidence: {lgb_proba:.4f} ({lgb_proba*100:.2f}%)")
    
    # Ensemble prediction (average)
    ensemble_proba = (cb_proba + lgb_proba) / 2
    ensemble_pred = 1 if ensemble_proba > 0.5 else 0
    
    print(f"\nEnsemble (Average):")
    print(f"  Prediction: {'PRIVILEGE ESCALATION' if ensemble_pred == 1 else 'NORMAL'}")
    print(f"  Confidence: {ensemble_proba:.4f} ({ensemble_proba*100:.2f}%)")

print("\n" + "="*80)
print("✓ Predictions completed successfully!")
print("="*80 + "\n")
