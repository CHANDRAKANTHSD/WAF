# LDAP Injection Detection WAF - Dual Implementation

This project implements two state-of-the-art Web Application Firewall (WAF) systems for LDAP injection detection:

1. **XGBoost-based WAF** - Traditional ML with feature engineering
2. **CNN-BiLSTM with Attention** - Deep learning with sequence processing

Both models train consecutively on three datasets:
- CICDDoS2019 LDAP Dataset
- LSNM2024 Dataset (Benign + Malicious)
- CSIC Database

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train XGBoost Model

```bash
python xgboost_waf_ldap.py
```

**Features:**
- 80+ network feature extraction
- SMOTE for class imbalance handling
- 70/15/15 train/val/test split
- Hyperparameter tuning (max_depth=6, learning_rate=0.1, n_estimators=200)
- Feature importance analysis
- Confusion matrix and ROC-AUC visualization
- Real-time prediction function
- Model saved as `xgboost_waf_model.pkl`

**Outputs:**
- `xgboost_waf_model.pkl` - Trained model
- `confusion_matrix_*_xgboost.png` - Confusion matrices for each dataset
- `roc_curve_*_xgboost.png` - ROC curves
- `feature_importance_xgboost.png` - Feature importance plot

### Train CNN-BiLSTM Model

```bash
python cnn_bilstm_waf_ldap.py
```

**Features:**
- Character-level tokenization and padding
- Dual CNN channels (filters=128, kernels=3,5)
- BiLSTM layer (units=64)
- Custom attention mechanism
- Adam optimizer with binary cross-entropy loss
- Early stopping and learning rate reduction
- Attention weight visualization
- Real-time inference pipeline
- Model saved as `cnn_bilstm_waf_model.h5`

**Outputs:**
- `cnn_bilstm_waf_model.h5` - Trained Keras model
- `cnn_bilstm_tokenizer.pkl` - Tokenizer and metadata
- `confusion_matrix_*_cnn_bilstm.png` - Confusion matrices
- `training_history_*_cnn_bilstm.png` - Training curves
- `attention_weights_*_cnn_bilstm.png` - Attention visualizations

## Model Architecture

### XGBoost WAF
```
Input Features (80+) → StandardScaler → SMOTE → XGBoost Classifier
                                                  ├─ max_depth: 6
                                                  ├─ learning_rate: 0.1
                                                  └─ n_estimators: 200
```

### CNN-BiLSTM WAF
```
Input Text → Tokenization → Padding → Embedding(128)
                                      ├─ CNN(filters=128, kernel=3)
                                      ├─ CNN(filters=128, kernel=5)
                                      └─ Concatenate
                                         ↓
                                      BiLSTM(64)
                                         ↓
                                      Attention Layer
                                         ↓
                                      Dense(64) → Dense(32) → Dense(1, sigmoid)
```

## Real-Time Prediction

### XGBoost
```python
from xgboost_waf_ldap import XGBoostWAF

# Load model
waf = XGBoostWAF.load_model('xgboost_waf_model.pkl')

# Predict
query_features = {...}  # Feature dictionary
result = waf.predict_realtime(query_features)
print(result)
# {'is_attack': True, 'attack_probability': 0.95, 'benign_probability': 0.05}
```

### CNN-BiLSTM
```python
from cnn_bilstm_waf_ldap import CNNBiLSTMWAF

# Load model
waf = CNNBiLSTMWAF.load_model('cnn_bilstm_waf_model.h5', 'cnn_bilstm_tokenizer.pkl')

# Predict
query_text = "ldap://server/cn=*)(uid=*))(|(cn=*"
result = waf.predict_realtime(query_text)
print(result)
# {'is_attack': True, 'attack_probability': 0.98, 'benign_probability': 0.02}
```

## Evaluation Metrics

Both models report:
- **Accuracy** - Overall correctness
- **Precision** - True positives / (True positives + False positives)
- **Recall** - True positives / (True positives + False negatives)
- **F1-Score** - Harmonic mean of precision and recall
- **False Positive Rate** - False alarms rate
- **ROC-AUC** - Area under ROC curve (XGBoost only)

## Dataset Information

### CICDDoS2019 LDAP
- Format: Parquet
- Features: 78 network flow features
- Classes: NetBIOS (benign) vs LDAP attacks

### LSNM2024
- Format: CSV
- Features: 60+ packet-level features
- Classes: Normal vs Fuzzing/SQL Injection

### CSIC
- Format: CSV
- Features: HTTP request components
- Classes: Normal (0) vs Anomalous (1)

## Production Deployment

Both models are ready for production deployment:

1. **XGBoost Model** - Lightweight, fast inference, interpretable
2. **CNN-BiLSTM Model** - High accuracy, handles complex patterns

Choose based on your requirements:
- Use XGBoost for speed and interpretability
- Use CNN-BiLSTM for maximum detection accuracy

## File Structure

```
.
├── xgboost_waf_ldap.py              # XGBoost implementation
├── cnn_bilstm_waf_ldap.py           # CNN-BiLSTM implementation
├── requirements.txt                  # Dependencies
├── README.md                         # This file
├── cicddos_2019/                     # CICDDoS2019 dataset
├── LSNM2024 Dataset/                 # LSNM2024 dataset
├── csic_database.csv                 # CSIC dataset
└── [Generated files]
    ├── xgboost_waf_model.pkl
    ├── cnn_bilstm_waf_model.h5
    ├── cnn_bilstm_tokenizer.pkl
    └── [Visualization plots]
```

## Notes

- Training time varies based on hardware (GPU recommended for CNN-BiLSTM)
- Models train consecutively on all datasets
- Each dataset evaluation is independent
- Models can be fine-tuned by adjusting hyperparameters
- Both models handle class imbalance (SMOTE for XGBoost, class weights for CNN-BiLSTM)
