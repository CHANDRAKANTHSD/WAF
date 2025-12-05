# Quick Start Guide

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Train the Model

```bash
python ddos_detection.py
```

This will:
- Process all 7 datasets sequentially
- Compare LightGBM vs XGBoost
- Select the best model
- Save trained model files

**Expected Output:**
- Model comparison metrics for each dataset
- Feature importance plots
- Final best model selection
- Saved model files (.pkl and .joblib)

**Training Time:** Approximately 10-30 minutes depending on your hardware

## Step 3: Use the Trained Model

### Option A: Using the Inference Script

```bash
python inference.py
```

### Option B: Integrate in Your Application

```python
from inference import DDoSInference

# Initialize
detector = DDoSInference(
    model_path='LightGBM_ddos_model.joblib',  # or XGBoost
    scaler_path='scaler.joblib',
    features_path='selected_features.pkl'
)

# Predict on DataFrame
predictions, probabilities, time_ms = detector.predict(df)

# Predict single sample
result = detector.predict_single(sample_dict)
print(f"Result: {result['prediction']}")
print(f"Confidence: {result['confidence']*100:.2f}%")
```

## Troubleshooting

### Issue: File not found
- Ensure dataset paths in `ddos_detection.py` match your actual file locations
- Update the paths in the `datasets` list (lines 200-212)

### Issue: Memory error
- Reduce the number of features: Change `n_features=30` to a lower value
- Process datasets one at a time by commenting out others

### Issue: SMOTE error (k_neighbors)
- This happens with very small datasets
- The script automatically adjusts k_neighbors based on minority class size

## Model Files

After training, you'll have:
- `LightGBM_ddos_model.pkl` or `XGBoost_ddos_model.pkl`
- `LightGBM_ddos_model.joblib` or `XGBoost_ddos_model.joblib`
- `scaler.joblib`
- `selected_features.pkl`
- `*_feature_importance.png`

Use `.joblib` files for production (faster loading).

## Performance Expectations

Typical results:
- **Accuracy**: 95-99%
- **F1-Score**: 0.95-0.99
- **Inference Time**: 0.1-1 ms per sample
- **Model Size**: 1-5 MB

## Next Steps

1. Integrate the model into your WAF application
2. Set up real-time monitoring
3. Implement alerting based on predictions
4. Periodically retrain with new data
