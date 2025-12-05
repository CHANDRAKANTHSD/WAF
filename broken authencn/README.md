# WAF Broken Authentication Detection System

A comprehensive Web Application Firewall (WAF) model to detect broken authentication attacks using XGBoost and LSTM with sequential fine-tuning.

## Features

- **Sequential Fine-tuning**: Progressive training across three datasets
  1. Mobile Security Dataset
  2. Cybersecurity Attack Dataset  
  3. RBA (Risk-Based Authentication) Dataset

- **Dual Algorithm Implementation**: XGBoost and LSTM comparison
- **Imbalanced Data Handling**: SMOTE oversampling and class weights
- **Real-time Inference**: Optimized for <100ms prediction latency
- **Comprehensive Metrics**: Precision, Recall, F1-Score, AUC-ROC

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train Models with Sequential Fine-tuning

```bash
python waf_model.py
```

This will:
- Load and preprocess all three datasets
- Perform sequential fine-tuning (Stage 1 → Stage 2 → Stage 3)
- Train both XGBoost and LSTM models
- Handle imbalanced data using SMOTE
- Evaluate and compare models
- Save trained models and performance report

### 2. Visualize Results

```bash
python visualize_results.py
```

Generates:
- Model comparison charts
- Performance metrics visualization
- Latency analysis
- Summary table (CSV)

### 3. Real-time Inference Demo

```bash
python realtime_inference.py
```

Demonstrates:
- Single prediction inference
- Batch prediction capabilities
- Latency measurements
- Throughput analysis

## Model Architecture

### XGBoost
- Gradient boosting with histogram-based tree construction
- Class weight balancing
- Early stopping on validation set
- Optimized for speed and accuracy

### LSTM
- 2-layer LSTM architecture (64 → 32 units)
- Dropout regularization (0.3)
- Dense output layer with sigmoid activation
- Adam optimizer with learning rate 0.001

## Features Extracted

- `login_attempts`: Number of login attempts per user
- `failed_attempts`: Number of failed login attempts
- `session_duration`: Session duration or RTT proxy
- `ip_changes`: Number of unique IP addresses per user
- `device_type`: Device type (mobile, desktop, tablet, etc.)
- `hour`: Hour of login attempt (0-23)
- `day_of_week`: Day of week (0-6)

## Performance Metrics

Models are evaluated on:
- **Precision**: Accuracy of attack predictions
- **Recall**: Coverage of actual attacks
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve
- **Latency**: Inference time per prediction (target: <100ms)

## Sequential Training Pipeline

```
Stage 1: Mobile Security Dataset
    ↓ (Transfer Learning)
Stage 2: Cybersecurity Attack Dataset
    ↓ (Fine-tuning)
Stage 3: RBA Dataset
    ↓
Final Model
```

## Output Files

- `xgboost_model.json`: Trained XGBoost model
- `lstm_model.h5`: Trained LSTM model
- `scaler.pkl`: Feature scaler
- `encoders.pkl`: Label encoders
- `performance_report.json`: Detailed performance metrics
- `model_comparison.png`: Visualization charts
- `performance_summary.csv`: Summary table

## Real-time Inference API

```python
from realtime_inference import RealtimeWAFDetector

# Initialize detector
detector = RealtimeWAFDetector(model_type='xgboost')
detector.load_model()

# Make prediction
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
print(f"Prediction: {result['is_attack']}")
print(f"Risk Level: {result['risk_level']}")
print(f"Latency: {result['latency_ms']:.2f} ms")
```

## Model Selection Criteria

The best model is selected based on:
- F1-Score (40%)
- AUC-ROC (40%)
- Inference Latency (20%, inverted)

## Requirements

- Python 3.8+
- pandas 2.1.4
- numpy 1.26.2
- scikit-learn 1.3.2
- xgboost 2.0.3
- tensorflow 2.15.0
- imbalanced-learn 0.11.0
- matplotlib 3.8.2
- seaborn 0.13.0

## License

MIT License
