import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("WAF Privilege Escalation Detection - CatBoost & LightGBM Models")
print("="*80)

# ============================================================================
# STEP 1: Load and Process All Three Datasets
# ============================================================================

print("\n[1/6] Loading datasets...")

# Dataset 1: Attack_Dataset.csv
df1 = pd.read_csv('Attack_Dataset.csv')
print(f"✓ Attack_Dataset.csv loaded: {df1.shape}")

# Dataset 2: CLOUD_VULRABILITES_DATASET.jsonl
data_list = []
with open('CLOUD_VULRABILITES_DATASET.jsonl', 'r') as f:
    for line in f:
        data_list.append(json.loads(line))
df2 = pd.DataFrame(data_list)
print(f"✓ CLOUD_VULRABILITES_DATASET.jsonl loaded: {df2.shape}")

# Dataset 3: embedded_system_network_security_dataset.csv
df3 = pd.read_csv('embedded_system_network_security_dataset.csv')
print(f"✓ embedded_system_network_security_dataset.csv loaded: {df3.shape}")

# ============================================================================
# STEP 2: Feature Engineering and Preprocessing
# ============================================================================

print("\n[2/6] Processing and combining datasets...")

# Process Dataset 1 (Attack_Dataset)
df1_processed = df1.copy()
df1_processed['dataset_source'] = 'attack_dataset'

# Create target variable for privilege escalation detection
privilege_escalation_keywords = ['privilege escalation', 'admin', 'root', 'escalation', 
                                   'PassRole', 'sudo', 'administrator']
df1_processed['target'] = df1_processed.apply(
    lambda row: 1 if any(keyword.lower() in str(row).lower() for keyword in privilege_escalation_keywords) else 0,
    axis=1
)

# Extract categorical features
df1_cat_features = ['Category', 'Attack Type', 'Target Type', 'MITRE Technique']
df1_features = df1_processed[df1_cat_features + ['target']].copy()
df1_features.columns = ['attack_category', 'attack_type', 'target_system', 'mitre_technique', 'target']

# Process Dataset 2 (Cloud Vulnerabilities)
df2_processed = df2.copy()
df2_processed['dataset_source'] = 'cloud_vulnerabilities'

# Create target for privilege escalation
df2_processed['target'] = df2_processed.apply(
    lambda row: 1 if any(keyword in str(row).lower() for keyword in 
                         ['privilege', 'escalation', 'iam', 'passrole', 'admin', 'root']) else 0,
    axis=1
)

# Extract features
df2_features = pd.DataFrame({
    'attack_category': df2_processed['category'],
    'attack_type': df2_processed['category'],
    'target_system': df2_processed['cloud_provider'],
    'mitre_technique': 'Cloud_' + df2_processed['category'].astype(str),
    'target': df2_processed['target']
})

# Process Dataset 3 (Embedded System Network Security)
df3_processed = df3.copy()
df3_processed['dataset_source'] = 'embedded_network'

# Use existing label as target
df3_features = pd.DataFrame({
    'attack_category': 'Network_Security',
    'attack_type': 'Network_Attack',
    'target_system': 'Embedded_System',
    'mitre_technique': 'Network_Intrusion',
    'packet_size': df3_processed['packet_size'],
    'inter_arrival_time': df3_processed['inter_arrival_time'],
    'packet_count_5s': df3_processed['packet_count_5s'],
    'mean_packet_size': df3_processed['mean_packet_size'],
    'spectral_entropy': df3_processed['spectral_entropy'],
    'frequency_band_energy': df3_processed['frequency_band_energy'],
    'target': df3_processed['label']
})

print(f"✓ Dataset 1 processed: {df1_features.shape}, Positive samples: {df1_features['target'].sum()}")
print(f"✓ Dataset 2 processed: {df2_features.shape}, Positive samples: {df2_features['target'].sum()}")
print(f"✓ Dataset 3 processed: {df3_features.shape}, Positive samples: {df3_features['target'].sum()}")

# Combine all datasets
combined_df = pd.concat([df1_features, df2_features, df3_features], ignore_index=True)

# Fill missing numerical features with 0
numerical_cols = ['packet_size', 'inter_arrival_time', 'packet_count_5s', 
                  'mean_packet_size', 'spectral_entropy', 'frequency_band_energy']
for col in numerical_cols:
    if col not in combined_df.columns:
        combined_df[col] = 0.0
    else:
        combined_df[col] = combined_df[col].fillna(0.0)

# Fill missing categorical features
categorical_cols = ['attack_category', 'attack_type', 'target_system', 'mitre_technique']
for col in categorical_cols:
    combined_df[col] = combined_df[col].fillna('Unknown').astype(str)

print(f"\n✓ Combined dataset: {combined_df.shape}")
print(f"✓ Total positive samples: {combined_df['target'].sum()} ({combined_df['target'].mean()*100:.2f}%)")

# ============================================================================
# STEP 3: Prepare Features and Target
# ============================================================================

print("\n[3/6] Preparing features and splitting data...")

# Define feature columns
cat_features = ['attack_category', 'attack_type', 'target_system', 'mitre_technique']
num_features = ['packet_size', 'inter_arrival_time', 'packet_count_5s', 
                'mean_packet_size', 'spectral_entropy', 'frequency_band_energy']

X = combined_df[cat_features + num_features]
y = combined_df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Training set: {X_train.shape}, Positive: {y_train.sum()}")
print(f"✓ Test set: {X_test.shape}, Positive: {y_test.sum()}")

# Calculate class weights
class_counts = y_train.value_counts()
scale_pos_weight = class_counts[0] / class_counts[1] if 1 in class_counts.index else 1.0
print(f"✓ Class weight ratio: {scale_pos_weight:.2f}")

# ============================================================================
# STEP 4: Train CatBoost Model with Fine-tuning
# ============================================================================

print("\n[4/6] Training CatBoost model...")

# Create CatBoost Pool for better performance
train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool = Pool(X_test, y_test, cat_features=cat_features)

# Fine-tuned CatBoost parameters
catboost_params = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 8,
    'l2_leaf_reg': 3,
    'border_count': 128,
    'auto_class_weights': 'Balanced',
    'random_seed': 42,
    'verbose': 100,
    'early_stopping_rounds': 50,
    'task_type': 'CPU',
    'loss_function': 'Logloss',
    'eval_metric': 'AUC'
}

catboost_model = CatBoostClassifier(**catboost_params)
catboost_model.fit(train_pool, eval_set=test_pool, use_best_model=True)

print("\n✓ CatBoost training completed!")

# ============================================================================
# STEP 5: Train LightGBM Model with Fine-tuning
# ============================================================================

print("\n[5/6] Training LightGBM model...")

# Encode categorical features for LightGBM
X_train_lgb = X_train.copy()
X_test_lgb = X_test.copy()

label_encoders = {}
for col in cat_features:
    le = LabelEncoder()
    # Fit on all unique values from both train and test
    all_values = pd.concat([X_train[col], X_test[col]]).unique()
    le.fit(all_values.astype(str))
    X_train_lgb[col] = le.transform(X_train_lgb[col].astype(str))
    X_test_lgb[col] = le.transform(X_test_lgb[col].astype(str))
    label_encoders[col] = le

# Fine-tuned LightGBM parameters with better handling of imbalanced data
lightgbm_params = {
    'n_estimators': 500,
    'learning_rate': 0.1,
    'max_depth': 6,
    'num_leaves': 31,
    'min_child_samples': 10,
    'min_child_weight': 0.001,
    'subsample': 0.8,
    'subsample_freq': 1,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'scale_pos_weight': scale_pos_weight,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc'
}

lightgbm_model = LGBMClassifier(**lightgbm_params)
lightgbm_model.fit(
    X_train_lgb, y_train,
    eval_set=[(X_test_lgb, y_test)],
    eval_metric='auc',
    callbacks=[
        __import__('lightgbm').early_stopping(stopping_rounds=50),
        __import__('lightgbm').log_evaluation(period=100)
    ]
)

print("\n✓ LightGBM training completed!")

# ============================================================================
# STEP 6: Evaluate Both Models
# ============================================================================

print("\n[6/6] Evaluating models...")

def evaluate_model(model, X_test_data, y_test_data, model_name):
    """Evaluate model and print detailed metrics"""
    print(f"\n{'='*80}")
    print(f"{model_name} - Evaluation Results")
    print(f"{'='*80}")
    
    # Predictions
    y_pred = model.predict(X_test_data)
    y_pred_proba = model.predict_proba(X_test_data)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_data, y_pred)
    precision = precision_score(y_test_data, y_pred, zero_division=0)
    recall = recall_score(y_test_data, y_pred, zero_division=0)
    f1 = f1_score(y_test_data, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_test_data, y_pred_proba)
    except:
        roc_auc = 0.0
    
    # Print metrics
    print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test_data, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0][0]:6d}  |  FP: {cm[0][1]:6d}")
    print(f"  FN: {cm[1][0]:6d}  |  TP: {cm[1][1]:6d}")
    
    # Classification Report
    print(f"\nClassification Report:")
    print(classification_report(y_test_data, y_pred, zero_division=0))
    
    # Confidence scores statistics
    print(f"\nConfidence Score Statistics:")
    print(f"  Mean:   {y_pred_proba.mean():.4f}")
    print(f"  Median: {np.median(y_pred_proba):.4f}")
    print(f"  Std:    {y_pred_proba.std():.4f}")
    print(f"  Min:    {y_pred_proba.min():.4f}")
    print(f"  Max:    {y_pred_proba.max():.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

# Evaluate CatBoost
catboost_results = evaluate_model(catboost_model, X_test, y_test, "CatBoost")

# Evaluate LightGBM
lightgbm_results = evaluate_model(lightgbm_model, X_test_lgb, y_test, "LightGBM")

# ============================================================================
# STEP 7: Save Models
# ============================================================================

print(f"\n{'='*80}")
print("Saving Models")
print(f"{'='*80}")

# Save CatBoost model in multiple formats
catboost_model.save_model('model/catboost_waf_model.cbm')
print("✓ CatBoost model saved: model/catboost_waf_model.cbm")

# Save CatBoost as pickle
with open('model/catboost_waf_model.pkl', 'wb') as f:
    pickle.dump(catboost_model, f)
print("✓ CatBoost model saved: model/catboost_waf_model.pkl")

# Save LightGBM model
with open('model/lightgbm_waf_model.pkl', 'wb') as f:
    pickle.dump(lightgbm_model, f)
print("✓ LightGBM model saved: model/lightgbm_waf_model.pkl")

# Save LightGBM in native format
lightgbm_model.booster_.save_model('model/lightgbm_waf_model.txt')
print("✓ LightGBM model saved: model/lightgbm_waf_model.txt")

# Save label encoders for LightGBM
with open('model/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("✓ Label encoders saved: model/label_encoders.pkl")

# Save feature names
feature_info = {
    'categorical_features': cat_features,
    'numerical_features': num_features,
    'all_features': cat_features + num_features
}
with open('model/feature_info.pkl', 'wb') as f:
    pickle.dump(feature_info, f)
print("✓ Feature info saved: model/feature_info.pkl")

# ============================================================================
# Final Summary
# ============================================================================

print(f"\n{'='*80}")
print("Training Summary")
print(f"{'='*80}")
print(f"\nTotal samples processed: {len(combined_df)}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"\nCategorical features: {len(cat_features)}")
print(f"Numerical features: {len(num_features)}")
print(f"\nModel Comparison:")
print(f"  CatBoost  - Accuracy: {catboost_results['accuracy']:.4f}, F1: {catboost_results['f1_score']:.4f}, AUC: {catboost_results['roc_auc']:.4f}")
print(f"  LightGBM  - Accuracy: {lightgbm_results['accuracy']:.4f}, F1: {lightgbm_results['f1_score']:.4f}, AUC: {lightgbm_results['roc_auc']:.4f}")

print(f"\n{'='*80}")
print("✓ All tasks completed successfully!")
print(f"{'='*80}\n")
