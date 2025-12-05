# Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
```

## Training

Run the complete sequential fine-tuning pipeline:

```bash
python waf_model.py
```

This will:
1. Load all three datasets
2. Train XGBoost and LSTM models
3. Apply SMOTE for imbalanced data
4. Evaluate and compare models
5. Save trained models

**Expected Output**:
- Training time: ~2 minutes
- Best model: XGBoost
- F1-Score: 0.27
- AUC-ROC: 0.72
- Latency: 0.01 ms

## Visualization

Generate performance charts:

```bash
python visualize_results.py
```

**Output**: `model_comparison.png` with 4 charts

## Real-Time Inference

Test the trained models:

```bash
python realtime_inference.py
```

## Using the Model in Your Code

```python
from realtime_inference import RealtimeWAFDetector

# Initialize
detector = RealtimeWAFDetector(model_type='xgboost')
detector.load_model()

# Predict
sample = {
    'login_attempts': 50,
    'failed_attempts': 30,
    'session_duration': 120,
    'ip_changes': 8,
    'device_type': 'desktop',
    'hour': 3,
    'day_of_week': 6
}

result = detector.predict(sample)
print(f"Attack: {result['is_attack']}")
print(f"Risk: {result['risk_level']}")
print(f"Probability: {result['probability']:.2%}")
```

## Output Files

After training, you'll have:
- `xgboost_model.json` - XGBoost model
- `lstm_model.h5` - LSTM model
- `scaler.pkl` - Feature scaler
- `encoders.pkl` - Label encoders
- `performance_report.json` - Metrics
- `model_comparison.png` - Charts
- `performance_summary.csv` - Summary table

## Performance Summary

| Model | F1-Score | AUC-ROC | Latency | Status |
|-------|----------|---------|---------|--------|
| XGBoost | 0.2678 | 0.7191 | 0.01 ms | ✓ Winner |
| LSTM | 0.2302 | 0.6532 | 0.46 ms | ✓ Pass |

## Next Steps

1. Review `PERFORMANCE_REPORT.md` for detailed analysis
2. Adjust thresholds for your use case
3. Integrate into your WAF system
4. Monitor and retrain periodically
