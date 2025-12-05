import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report, confusion_matrix, accuracy_score, roc_auc_score
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# Feature engineering functions
def extract_features(payload):
    features = {}
    features['length'] = len(payload)
    features['num_special_chars'] = len(re.findall(r'[^a-zA-Z0-9\s]', payload))
    features['num_digits'] = len(re.findall(r'\d', payload))
    features['entropy'] = calculate_entropy(payload)
    features['has_sql_keywords'] = int(bool(re.search(r'(union|select|insert|update|delete|drop|exec)', payload, re.IGNORECASE)))
    features['has_xss_patterns'] = int(bool(re.search(r'(<script|javascript:|onerror|onload)', payload, re.IGNORECASE)))
    features['has_path_traversal'] = int(bool(re.search(r'(\.\./|\.\.\\)', payload)))
    features['url_encoded_chars'] = payload.count('%')
    return features

def calculate_entropy(text):
    if not text:
        return 0
    prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
    entropy = -sum([p * np.log2(p) for p in prob])
    return entropy

# Load data
def load_data():
    with open('archive (1)/WEB_APPLICATION_PAYLOADS.jsonl', 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        content = content.replace('\x00', '')
        try:
            json_data = json.loads(content)
        except:
            json_data = []
    
    payloads = []
    labels = []
    for item in json_data:
        if isinstance(item, dict):
            payloads.append(item.get('payload', ''))
            labels.append(1 if item.get('type') != 'benign' else 0)
    
    df_json = pd.DataFrame({'payload': payloads, 'label': labels})
    
    df_csv = pd.read_csv('archive (2)/csic_database.csv')
    
    if 'URL' in df_csv.columns:
        df_csv['payload'] = df_csv['URL'].fillna('') + ' ' + df_csv.get('content', '').fillna('')
    else:
        df_csv['payload'] = df_csv.iloc[:, -1].astype(str)
    
    if 'classification' in df_csv.columns:
        # Classification is already 0 (normal) or 1 (anomalous)
        df_csv['label'] = df_csv['classification']
    else:
        df_csv['label'] = 1
    
    df_csv = df_csv[['payload', 'label']]
    df = pd.concat([df_json, df_csv], ignore_index=True)
    
    return df

# Prepare data
def prepare_data(df):
    feature_list = []
    for payload in df['payload']:
        feature_list.append(extract_features(str(payload)))
    
    features_df = pd.DataFrame(feature_list)
    
    tokenizer = Tokenizer(num_words=10000, char_level=True)
    tokenizer.fit_on_texts(df['payload'].astype(str))
    sequences = tokenizer.texts_to_sequences(df['payload'].astype(str))
    X_seq = pad_sequences(sequences, maxlen=200)
    
    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    
    X_train_seq, X_test_seq, X_train_feat, X_test_feat, y_train, y_test = train_test_split(
        X_seq, features_df, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_feat_scaled = scaler.fit_transform(X_train_feat)
    X_test_feat_scaled = scaler.transform(X_test_feat)
    
    return X_train_seq, X_test_seq, X_train_feat_scaled, X_test_feat_scaled, y_train, y_test

print("="*80)
print("WAF ML PIPELINE - PERFORMANCE EVALUATION")
print("="*80)

print("\n[1/4] Loading data...")
df = load_data()
print(f"✓ Total samples: {len(df)}")
print(f"  - Malicious: {df['label'].sum()} ({df['label'].sum()/len(df)*100:.1f}%)")
print(f"  - Benign: {len(df) - df['label'].sum()} ({(len(df) - df['label'].sum())/len(df)*100:.1f}%)")

print("\n[2/4] Preparing test data...")
X_train_seq, X_test_seq, X_train_feat, X_test_feat, y_train, y_test = prepare_data(df)
print(f"✓ Test set size: {len(y_test)} samples")

print("\n[3/4] Loading trained models...")
lgb_model = lgb.Booster(model_file='lgb_waf_model.txt')
lstm_model = load_model('lstm_waf_model.h5')
print("✓ Models loaded successfully")

print("\n[4/4] Evaluating models...")
print("\n" + "="*80)

# LightGBM predictions
print("\n📊 LIGHTGBM MODEL PERFORMANCE:")
print("-"*80)
lgb_pred_proba = lgb_model.predict(X_test_feat)
lgb_pred = (lgb_pred_proba > 0.5).astype(int)

print(f"Accuracy:  {accuracy_score(y_test, lgb_pred):.4f}")
print(f"Precision: {precision_score(y_test, lgb_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, lgb_pred, zero_division=0):.4f}")
print(f"F1-Score:  {f1_score(y_test, lgb_pred, zero_division=0):.4f}")
try:
    print(f"ROC-AUC:   {roc_auc_score(y_test, lgb_pred_proba):.4f}")
except:
    pass

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, lgb_pred)
print(f"  True Negatives:  {cm[0][0] if cm.shape[0] > 1 else 0}")
print(f"  False Positives: {cm[0][1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0}")
print(f"  False Negatives: {cm[1][0] if cm.shape[0] > 1 else 0}")
print(f"  True Positives:  {cm[1][1] if cm.shape[0] > 1 and cm.shape[1] > 1 else 0}")

# LSTM predictions
print("\n" + "="*80)
print("\n📊 LSTM MODEL PERFORMANCE:")
print("-"*80)
lstm_pred_proba = lstm_model.predict(X_test_seq, verbose=0).flatten()
lstm_pred = (lstm_pred_proba > 0.5).astype(int)

print(f"Accuracy:  {accuracy_score(y_test, lstm_pred):.4f}")
print(f"Precision: {precision_score(y_test, lstm_pred):.4f}")
print(f"Recall:    {recall_score(y_test, lstm_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, lstm_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, lstm_pred_proba):.4f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, lstm_pred)
print(f"  True Negatives:  {cm[0][0]}")
print(f"  False Positives: {cm[0][1]}")
print(f"  False Negatives: {cm[1][0]}")
print(f"  True Positives:  {cm[1][1]}")

# Ensemble predictions
print("\n" + "="*80)
print("\n📊 ENSEMBLE MODEL PERFORMANCE (50% LightGBM + 50% LSTM):")
print("-"*80)
ensemble_pred_proba = (0.5 * lgb_pred_proba + 0.5 * lstm_pred_proba)
ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)

print(f"Accuracy:  {accuracy_score(y_test, ensemble_pred):.4f}")
print(f"Precision: {precision_score(y_test, ensemble_pred):.4f}")
print(f"Recall:    {recall_score(y_test, ensemble_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, ensemble_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, ensemble_pred_proba):.4f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, ensemble_pred)
print(f"  True Negatives:  {cm[0][0]}")
print(f"  False Positives: {cm[0][1]}")
print(f"  False Negatives: {cm[1][0]}")
print(f"  True Positives:  {cm[1][1]}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, ensemble_pred, target_names=['Benign', 'Malicious']))

print("\n" + "="*80)
print("🎯 SUMMARY:")
print("-"*80)
print(f"Best Model: {'LSTM' if f1_score(y_test, lstm_pred) > f1_score(y_test, ensemble_pred) else 'Ensemble'}")
print(f"Total Attacks Detected: {ensemble_pred.sum()} / {y_test.sum()}")
print(f"Detection Rate: {recall_score(y_test, ensemble_pred)*100:.2f}%")
print(f"False Positive Rate: {cm[0][1]/(cm[0][0]+cm[0][1])*100:.2f}%")
print("="*80)
