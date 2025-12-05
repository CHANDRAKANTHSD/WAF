import pandas as pd
import numpy as np
import json
import re
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report, confusion_matrix
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load datasets
def load_data():
    # Load JSON payload data with error handling
    try:
        with open('archive (1)/WEB_APPLICATION_PAYLOADS.jsonl', 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            # Try to fix common JSON issues
            content = content.replace('\x00', '')  # Remove null bytes
            json_data = json.loads(content)
    except:
        # If JSON parsing fails, try reading line by line
        json_data = []
        with open('archive (1)/WEB_APPLICATION_PAYLOADS.jsonl', 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line and line not in ['{', '}', '[', ']', ',']:
                    try:
                        json_data.append(json.loads(line))
                    except:
                        pass
    
    payloads = []
    labels = []
    for item in json_data:
        if isinstance(item, dict):
            payloads.append(item.get('payload', ''))
            # Assume malicious if not explicitly marked as benign
            labels.append(1 if item.get('type') != 'benign' else 0)
    
    df_json = pd.DataFrame({'payload': payloads, 'label': labels})
    
    # Load CSV data
    df_csv = pd.read_csv('archive (2)/csic_database.csv')
    
    # Extract payload from CSV (combine relevant fields)
    if 'URL' in df_csv.columns:
        df_csv['payload'] = df_csv['URL'].fillna('') + ' ' + df_csv.get('content', '').fillna('')
    else:
        df_csv['payload'] = df_csv.iloc[:, -1].astype(str)  # Use last column as payload
    
    # Map classification to binary label
    if 'classification' in df_csv.columns:
        # Classification is already 0 (normal) or 1 (anomalous)
        df_csv['label'] = df_csv['classification']
    else:
        df_csv['label'] = 1  # Assume malicious if no classification
    
    df_csv = df_csv[['payload', 'label']]
    
    # Combine datasets
    df = pd.concat([df_json, df_csv], ignore_index=True)
    
    return df

# Feature engineering
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

# Prepare data
def prepare_data(df):
    print("  → Extracting features from payloads...")
    # Extract engineered features
    feature_list = []
    for i, payload in enumerate(df['payload']):
        if i % 10000 == 0 and i > 0:
            print(f"    Processed {i}/{len(df)} payloads...")
        feature_list.append(extract_features(str(payload)))
    
    features_df = pd.DataFrame(feature_list)
    print(f"  → Feature extraction complete: {features_df.shape[1]} features")
    
    print("  → Tokenizing payloads for LSTM...")
    # Tokenize payloads for deep learning
    tokenizer = Tokenizer(num_words=10000, char_level=True)
    tokenizer.fit_on_texts(df['payload'].astype(str))
    sequences = tokenizer.texts_to_sequences(df['payload'].astype(str))
    X_seq = pad_sequences(sequences, maxlen=200)
    print(f"  → Tokenization complete: sequences shape {X_seq.shape}")
    
    # Prepare labels
    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    
    print("  → Splitting data (80% train, 20% test)...")
    # Split data
    X_train_seq, X_test_seq, X_train_feat, X_test_feat, y_train, y_test = train_test_split(
        X_seq, features_df, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("  → Scaling features...")
    # Scale features
    scaler = StandardScaler()
    X_train_feat_scaled = scaler.fit_transform(X_train_feat)
    X_test_feat_scaled = scaler.transform(X_test_feat)
    
    return X_train_seq, X_test_seq, X_train_feat_scaled, X_test_feat_scaled, y_train, y_test, tokenizer, scaler, le

# LightGBM model
def train_lightgbm(X_train, y_train, X_test, y_test):
    print("\n=== Training LightGBM ===")
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1
    }
    
    model = lgb.train(params, train_data, num_boost_round=100, valid_sets=[test_data])
    
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model

# LSTM model
def train_lstm(X_train_seq, y_train, X_test_seq, y_test):
    print("\n=== Training LSTM ===")
    model = Sequential([
        Embedding(input_dim=10000, output_dim=128, input_length=200),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001)
    
    history = model.fit(
        X_train_seq, y_train,
        validation_data=(X_test_seq, y_test),
        epochs=20,
        batch_size=64,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    y_pred = (model.predict(X_test_seq) > 0.5).astype(int).flatten()
    
    print(f"\nPrecision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model

# Ensemble prediction
def ensemble_predict(lgb_model, lstm_model, X_feat, X_seq, weights=[0.5, 0.5]):
    lgb_pred = lgb_model.predict(X_feat)
    lstm_pred = lstm_model.predict(X_seq).flatten()
    
    ensemble_pred = (weights[0] * lgb_pred + weights[1] * lstm_pred > 0.5).astype(int)
    return ensemble_pred

# Main pipeline
def main():
    print("="*70)
    print("WAF ML PIPELINE - TRAINING STARTED")
    print("="*70)
    
    print("\n[STEP 1/6] Loading data...")
    df = load_data()
    print(f"✓ Total samples loaded: {len(df)}")
    print(f"  - Malicious samples: {df['label'].sum()}")
    print(f"  - Benign samples: {len(df) - df['label'].sum()}")
    
    print("\n[STEP 2/6] Preparing data (feature extraction & tokenization)...")
    X_train_seq, X_test_seq, X_train_feat, X_test_feat, y_train, y_test, tokenizer, scaler, le = prepare_data(df)
    print(f"✓ Data prepared:")
    print(f"  - Training samples: {len(X_train_seq)}")
    print(f"  - Test samples: {len(X_test_seq)}")
    print(f"  - Features extracted: {X_train_feat.shape[1]}")
    
    # Train models
    print("\n[STEP 3/6] Training LightGBM model...")
    lgb_model = train_lightgbm(X_train_feat, y_train, X_test_feat, y_test)
    print("✓ LightGBM training completed")
    
    print("\n[STEP 4/6] Training LSTM model (this may take several minutes)...")
    lstm_model = train_lstm(X_train_seq, y_train, X_test_seq, y_test)
    print("✓ LSTM training completed")
    
    # Ensemble evaluation
    print("\n[STEP 5/6] Evaluating Ensemble Model...")
    print("="*70)
    ensemble_pred = ensemble_predict(lgb_model, lstm_model, X_test_feat, X_test_seq)
    
    print(f"\n📊 ENSEMBLE MODEL RESULTS:")
    print(f"  - Precision: {precision_score(y_test, ensemble_pred):.4f}")
    print(f"  - Recall: {recall_score(y_test, ensemble_pred):.4f}")
    print(f"  - F1-Score: {f1_score(y_test, ensemble_pred):.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, ensemble_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, ensemble_pred))
    
    # Save models
    print("\n[STEP 6/6] Saving models...")
    lgb_model.save_model('lgb_waf_model.txt')
    lstm_model.save('lstm_waf_model.h5')
    print("✓ Models saved:")
    print("  - lgb_waf_model.txt")
    print("  - lstm_waf_model.h5")
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)

if __name__ == "__main__":
    main()