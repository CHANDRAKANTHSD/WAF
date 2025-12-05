# WAF Privilege Escalation Detection using CatBoost & LightGBM

A comprehensive machine learning solution for detecting privilege escalation attacks in Web Application Firewalls (WAF) using CatBoost and LightGBM models trained on three diverse Kaggle datasets.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)

## 🎯 Overview

This project implements state-of-the-art gradient boosting models (CatBoost and LightGBM) to detect privilege escalation attacks across multiple domains:
- General cybersecurity attacks
- Cloud infrastructure vulnerabilities
- Embedded system network security

**Key Highlights:**
- ✅ Sequential training on 16,333 samples from 3 datasets
- ✅ Native categorical feature handling
- ✅ Automatic class weight balancing for imbalanced data
- ✅ 86.89% ROC-AUC score achieved
- ✅ Production-ready models with confidence scores
- ✅ Multiple model formats (.pkl, .cbm, .txt)

## 🚀 Features

### Model Capabilities
- **CatBoost Model**: 83.75% accuracy, 66.17% recall, 84.83% ROC-AUC
- **LightGBM Model**: 88.34% accuracy, 62.57% recall, 86.89% ROC-AUC
- **Ensemble Support**: Combine both models for maximum reliability

### Feature Engineering
**Categorical Features (Native Handling):**
- Attack category
- Attack type
- Target system
- MITRE ATT&CK technique

**Numerical Features:**
- Packet size
- Inter-arrival time
- Packet count (5s window)
- Mean packet size
- Spectral entropy
- Frequency band energy

## 📊 Datasets

### 1. Attack_Dataset.csv
- **Size**: 14,133 records
- **Domain**: General cybersecurity attacks
- **Positive Samples**: 1,430 privilege escalation cases

### 2. CLOUD_VULRABILITES_DATASET.jsonl
- **Size**: 1,200 records
- **Domain**: Cloud infrastructure vulnerabilities
- **Positive Samples**: 140 privilege escalation cases

### 3. embedded_system_network_security_dataset.csv
- **Size**: 1,000 records
- **Domain**: Embedded systems and network security
- **Positive Samples**: 100 attack cases

**Combined Dataset:**
- Total: 16,333 records
- Training: 13,066 samples (80%)
- Testing: 3,267 samples (20%)
- Positive Class: 10.22%

## 💻 Installation

### Prerequisites
```bash
Python 3.8+
```

### Install Dependencies
```bash
pip install pandas numpy scikit-learn catboost lightgbm
```

### Clone Repository
```bash
git clone <repository-url>
cd waf-privilege-escalation-detection
```

## 🔧 Usage

### 1. Train Models
Train both CatBoost and LightGBM models on all three datasets:

```bash
python waf_privilege_escalation_detection.py
```

**Output:**
- Trained models saved in `model/` directory
- Training metrics and evaluation results
- Confusion matrices and classification reports

### 2. Load and Predict
Load trained models and make predictions:

```bash
python load_and_predict.py
```

**Example Code:**
```python
from catboost import CatBoostClassifier
import pandas as pd

# Load model
model = CatBoostClassifier()
model.load_model('model/catboost_waf_model.cbm')

# Prepare sample
sample = pd.DataFrame([{
    'attack_category': 'IAM Misconfiguration',
    'attack_type': 'Privilege Escalation',
    'target_system': 'AWS',
    'mitre_technique': 'T1078',
    'packet_size': 0.5,
    'inter_arrival_time': 0.3,
    'packet_count_5s': 0.8,
    'mean_packet_size': 0.0,
    'spectral_entropy': 0.7,
    'frequency_band_energy': 0.6
}])

# Predict
prediction = model.predict(sample)
confidence = model.predict_proba(sample)[0][1]

print(f"Prediction: {'PRIVILEGE ESCALATION' if prediction[0] == 1 else 'NORMAL'}")
print(f"Confidence: {confidence:.4f}")
```

### 3. Generate Evaluation Report
Create comprehensive evaluation report:

```bash
python model_evaluation_report.py
```

**Outputs:**
- `model_evaluation_report.json` - Detailed JSON report
- `model_evaluation_report.txt` - Text summary

## 📈 Model Performance

### CatBoost Model
| Metric | Value |
|--------|-------|
| Accuracy | 83.75% |
| Precision | 34.59% |
| Recall | **66.17%** ⭐ |
| F1-Score | 45.43% |
| ROC-AUC | 84.83% |

**Confusion Matrix:**
```
TN: 2,515  |  FP: 418
FN: 113    |  TP: 221
```

### LightGBM Model
| Metric | Value |
|--------|-------|
| Accuracy | **88.34%** ⭐ |
| Precision | **44.95%** ⭐ |
| Recall | 62.57% |
| F1-Score | **52.32%** ⭐ |
| ROC-AUC | **86.89%** ⭐ |

**Confusion Matrix:**
```
TN: 2,677  |  FP: 256
FN: 125    |  TP: 209
```

### Model Comparison

| Aspect | CatBoost | LightGBM | Winner |
|--------|----------|----------|--------|
| Accuracy | 83.75% | **88.34%** | LightGBM |
| Precision | 34.59% | **44.95%** | LightGBM |
| Recall | **66.17%** | 62.57% | CatBoost |
| F1-Score | 45.43% | **52.32%** | LightGBM |
| ROC-AUC | 84.83% | **86.89%** | LightGBM |
| Training Time | **6s** | 18s | CatBoost |
| False Positives | 418 | **256** | LightGBM |
| False Negatives | **113** | 125 | CatBoost |

## 📁 Project Structure

```
waf-privilege-escalation-detection/
│
├── datasets/
│   ├── Attack_Dataset.csv
│   ├── CLOUD_VULRABILITES_DATASET.jsonl
│   └── embedded_system_network_security_dataset.csv
│
├── model/
│   ├── catboost_waf_model.cbm          # CatBoost native format (8.3 MB)
│   ├── catboost_waf_model.pkl          # CatBoost pickle format (8.3 MB)
│   ├── lightgbm_waf_model.pkl          # LightGBM pickle format (906 KB)
│   ├── lightgbm_waf_model.txt          # LightGBM text format (898 KB)
│   ├── label_encoders.pkl              # Label encoders for LightGBM
│   └── feature_info.pkl                # Feature metadata
│
├── waf_privilege_escalation_detection.py   # Main training script
├── load_and_predict.py                     # Inference script
├── model_evaluation_report.py              # Evaluation report generator
├── MODEL_SUMMARY.md                        # Detailed model documentation
├── model_evaluation_report.json            # JSON evaluation report
├── model_evaluation_report.txt             # Text evaluation report
└── README.md                               # This file
```

## 🎯 Results

### Key Achievements
✅ **Sequential Training**: Successfully trained on 3 diverse datasets without separate models  
✅ **Native Categorical Handling**: Both models handle categorical features efficiently  
✅ **Auto Class Weighting**: Automatic handling of imbalanced data (10.22% positive class)  
✅ **Fine-tuned Hyperparameters**: Optimized for privilege escalation detection  
✅ **Multiple Formats**: Models saved in .pkl, .cbm, and .txt formats  
✅ **High Performance**: 86.89% ROC-AUC with LightGBM  
✅ **Production Ready**: Complete inference pipeline with confidence scores  

### Recommendations

| Use Case | Recommended Model | Reason |
|----------|------------------|--------|
| Production Deployment | **LightGBM** | Better overall metrics (88.34% accuracy) |
| High Recall Scenarios | **CatBoost** | Fewer missed attacks (66.17% recall) |
| Balanced Approach | **Ensemble** | Combine both for maximum reliability |
| Real-time Inference | **CatBoost** | Faster prediction time |
| Batch Processing | **LightGBM** | Better accuracy and precision |

## 🔍 Technical Details

### Hyperparameters

**CatBoost:**
```python
{
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 8,
    'l2_leaf_reg': 3,
    'border_count': 128,
    'auto_class_weights': 'Balanced',
    'early_stopping_rounds': 50
}
```

**LightGBM:**
```python
{
    'n_estimators': 500,
    'learning_rate': 0.1,
    'max_depth': 6,
    'num_leaves': 31,
    'min_child_samples': 10,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 8.78
}
```

### Imbalanced Data Handling
- **Class Ratio**: 8.78:1 (negative:positive)
- **CatBoost**: Auto class weights
- **LightGBM**: Scale positive weight = 8.78

### Early Stopping
- **CatBoost**: Stopped at iteration 145/1000
- **LightGBM**: Stopped at iteration 293/500
- **Benefit**: Prevents overfitting, faster training

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions or support, please open an issue in the repository.

## 🙏 Acknowledgments

- MITRE ATT&CK Framework for attack categorization
- Kaggle for providing the datasets
- CatBoost and LightGBM teams for excellent ML libraries

---

**Last Updated**: November 22, 2025  
**Version**: 1.0.0  
**Status**: Production Ready ✅
