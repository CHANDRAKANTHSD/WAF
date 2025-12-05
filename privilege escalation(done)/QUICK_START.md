# Quick Start Guide - WAF Privilege Escalation Detection

## 🚀 Get Started in 3 Steps

### Step 1: Train the Models (5 minutes)
```bash
python waf_privilege_escalation_detection.py
```

**What happens:**
- Loads 3 datasets (16,333 total samples)
- Trains CatBoost model (~6 seconds)
- Trains LightGBM model (~18 seconds)
- Evaluates both models
- Saves models to `model/` directory

**Expected Output:**
```
CatBoost  - Accuracy: 83.75%, F1: 45.43%, AUC: 84.83%
LightGBM  - Accuracy: 88.34%, F1: 52.32%, AUC: 86.89%
```

---

### Step 2: Test the Models (1 minute)
```bash
python load_and_predict.py
```

**What happens:**
- Loads both trained models
- Makes predictions on 3 sample cases
- Shows confidence scores
- Demonstrates ensemble prediction

**Sample Output:**
```
Sample 1: IAM Misconfiguration
  CatBoost:  PRIVILEGE ESCALATION (54.19%)
  LightGBM:  PRIVILEGE ESCALATION (95.14%)
  Ensemble:  PRIVILEGE ESCALATION (74.66%)
```

---

### Step 3: Generate Report (30 seconds)
```bash
python model_evaluation_report.py
```

**What happens:**
- Creates comprehensive evaluation report
- Saves JSON and TXT reports
- Shows model comparison
- Provides recommendations

---

## 📊 Model Files Created

After training, you'll have these files in `model/` directory:

| File | Size | Description |
|------|------|-------------|
| `catboost_waf_model.cbm` | 8.3 MB | CatBoost native format |
| `catboost_waf_model.pkl` | 8.3 MB | CatBoost pickle format |
| `lightgbm_waf_model.pkl` | 906 KB | LightGBM pickle format |
| `lightgbm_waf_model.txt` | 898 KB | LightGBM text format |
| `label_encoders.pkl` | 70 MB | Label encoders for LightGBM |
| `feature_info.pkl` | 288 B | Feature metadata |

---

## 💡 Quick Usage Examples

### Example 1: Load CatBoost Model
```python
from catboost import CatBoostClassifier

model = CatBoostClassifier()
model.load_model('model/catboost_waf_model.cbm')
```

### Example 2: Load LightGBM Model
```python
import pickle

with open('model/lightgbm_waf_model.pkl', 'rb') as f:
    model = pickle.load(f)
```

### Example 3: Make Prediction
```python
import pandas as pd

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

prediction = model.predict(sample)
confidence = model.predict_proba(sample)[0][1]

print(f"Prediction: {prediction[0]}")
print(f"Confidence: {confidence:.4f}")
```

---

## 🎯 Which Model to Use?

| Scenario | Use This Model | Why? |
|----------|---------------|------|
| **Production** | LightGBM | Best overall accuracy (88.34%) |
| **Security Critical** | CatBoost | Highest recall (66.17%) - catches more attacks |
| **Real-time** | CatBoost | Faster inference |
| **Batch Processing** | LightGBM | Better precision (44.95%) |
| **Maximum Safety** | Ensemble | Combine both models |

---

## 📋 Required Features

When making predictions, provide these 10 features:

**Categorical (4):**
1. `attack_category` - e.g., "IAM Misconfiguration"
2. `attack_type` - e.g., "Privilege Escalation"
3. `target_system` - e.g., "AWS", "Azure", "GCP"
4. `mitre_technique` - e.g., "T1078"

**Numerical (6):**
5. `packet_size` - Range: 0.0 to 1.0
6. `inter_arrival_time` - Range: 0.0 to 1.0
7. `packet_count_5s` - Range: 0.0 to 1.0
8. `mean_packet_size` - Range: 0.0 to 1.0
9. `spectral_entropy` - Range: 0.0 to 1.0
10. `frequency_band_energy` - Range: 0.0 to 1.0

---

## 🔧 Troubleshooting

### Issue: "Module not found"
```bash
pip install pandas numpy scikit-learn catboost lightgbm
```

### Issue: "File not found"
Make sure you're in the project directory:
```bash
cd path/to/waf-privilege-escalation-detection
```

### Issue: "Model file corrupted"
Re-train the models:
```bash
python waf_privilege_escalation_detection.py
```

---

## 📈 Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **CatBoost** | 83.75% | 34.59% | **66.17%** | 45.43% | 84.83% |
| **LightGBM** | **88.34%** | **44.95%** | 62.57% | **52.32%** | **86.89%** |

**Winner:** LightGBM (better overall metrics)

---

## 🎓 Next Steps

1. ✅ Train models → `python waf_privilege_escalation_detection.py`
2. ✅ Test predictions → `python load_and_predict.py`
3. ✅ Generate report → `python model_evaluation_report.py`
4. 📖 Read detailed docs → `MODEL_SUMMARY.md`
5. 🚀 Deploy to production

---

## 📞 Need Help?

- 📖 Read `README.md` for detailed documentation
- 📊 Check `MODEL_SUMMARY.md` for technical details
- 📈 Review `model_evaluation_report.json` for metrics
- 💬 Open an issue on GitHub

---

**Last Updated**: November 22, 2025  
**Estimated Time**: 10 minutes total  
**Difficulty**: Beginner-friendly ✅
