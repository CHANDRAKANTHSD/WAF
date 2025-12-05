# Complete Guide - DDoS Detection System

## 📋 Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Training the Model](#training)
4. [Using the Model](#usage)
5. [API Deployment](#api)
6. [Integration](#integration)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This system trains and deploys a DDoS detection model using LightGBM and XGBoost on multiple datasets:
- **CICIDS2017** (3 files)
- **CSE-CIC-IDS2018** (3 files)
- **TON_IoT** (1 file)

The system automatically selects the best performing model and saves it for deployment.

---

## 📦 Installation

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
python test_setup.py
```

This will check:
- ✓ All required packages are installed
- ✓ Configuration is valid
- ✓ Dataset files are accessible

---

## 🎓 Training the Model

### Step 1: Configure Dataset Paths

Edit `config.py` and update the file paths to match your system:

```python
DATASETS = [
    {
        'path': r"D:\Your\Path\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        'name': "CICIDS2017-Friday"
    },
    # ... update all paths
]
```

### Step 2: Run Training

```bash
python ddos_detection.py
```

**What happens:**
1. Loads each dataset sequentially
2. Preprocesses and cleans data
3. Selects optimal features (top 30 by default)
4. Handles class imbalance with SMOTE
5. Trains both LightGBM and XGBoost
6. Compares performance metrics
7. Saves the best model

**Expected Duration:** 10-30 minutes

**Output Files:**
- `LightGBM_ddos_model.pkl` or `XGBoost_ddos_model.pkl`
- `LightGBM_ddos_model.joblib` or `XGBoost_ddos_model.joblib`
- `scaler.joblib`
- `selected_features.pkl`
- `*_feature_importance.png`

### Step 3: Review Results

The script displays:
- Accuracy, Precision, Recall, F1-Score for each dataset
- Inference time per sample
- Feature importance visualization
- Final model selection

---

## 🔮 Using the Model

### Option 1: Python Script

```python
from inference import DDoSInference

# Initialize
detector = DDoSInference(
    model_path='LightGBM_ddos_model.joblib',
    scaler_path='scaler.joblib',
    features_path='selected_features.pkl'
)

# Predict single sample
result = detector.predict_single(sample_dict)
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']*100:.2f}%")
```

### Option 2: Command Line

```bash
python inference.py
```

---

## 🚀 API Deployment

### Step 1: Start API Server

```bash
python api.py
```

Server runs on `http://localhost:5000`

### Step 2: Test API

In another terminal:

```bash
python test_api.py
```

### Step 3: Use API Endpoints

#### Health Check
```bash
curl http://localhost:5000/health
```

#### Single Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"flow_duration": 1000, "total_fwd_packets": 10}'
```

#### Batch Prediction
```bash
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"samples": [{"flow_duration": 1000}, {"flow_duration": 5000}]}'
```

#### Model Info
```bash
curl http://localhost:5000/model/info
```

---

## 🔗 Integration Examples

### JavaScript/Node.js

```javascript
const axios = require('axios');

async function detectDDoS(networkFeatures) {
  try {
    const response = await axios.post('http://localhost:5000/predict', networkFeatures);
    
    if (response.data.prediction === 'DDoS Attack') {
      console.log(`⚠️ DDoS Attack Detected! Confidence: ${response.data.confidence}`);
      // Trigger alert
    }
    
    return response.data;
  } catch (error) {
    console.error('Prediction error:', error);
  }
}
```

### Python

```python
import requests

def check_traffic(features):
    response = requests.post(
        'http://localhost:5000/predict',
        json=features
    )
    
    result = response.json()
    
    if result['prediction'] == 'DDoS Attack':
        print(f"⚠️ Alert! DDoS detected with {result['confidence']*100:.1f}% confidence")
        # Take action
    
    return result
```

### React Frontend

```javascript
import React, { useState } from 'react';

function DDoSDetector() {
  const [result, setResult] = useState(null);
  
  const checkTraffic = async (features) => {
    const response = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(features)
    });
    
    const data = await response.json();
    setResult(data);
  };
  
  return (
    <div>
      {result && (
        <div className={result.prediction === 'DDoS Attack' ? 'alert' : 'normal'}>
          <h3>{result.prediction}</h3>
          <p>Confidence: {(result.confidence * 100).toFixed(2)}%</p>
        </div>
      )}
    </div>
  );
}
```

---

## 🐛 Troubleshooting

### Issue: "Model not found"
**Solution:** Run training first: `python ddos_detection.py`

### Issue: "Dataset file not found"
**Solution:** Update paths in `config.py` to match your file locations

### Issue: "Memory error during training"
**Solution:** 
- Reduce `N_FEATURES` in `config.py` (e.g., from 30 to 20)
- Process fewer datasets at once
- Increase system RAM

### Issue: "SMOTE error: k_neighbors"
**Solution:** This is automatically handled. If it persists, check if dataset has enough samples.

### Issue: "API connection refused"
**Solution:** 
- Make sure API server is running: `python api.py`
- Check if port 5000 is available
- Try different port in `api.py`

### Issue: "Low accuracy"
**Solution:**
- Ensure datasets are properly labeled
- Check for data quality issues
- Try adjusting hyperparameters in `config.py`
- Retrain with more data

### Issue: "Slow predictions"
**Solution:**
- Use `.joblib` files instead of `.pkl`
- Reduce number of features
- Use batch predictions for multiple samples
- Consider model quantization

---

## 📊 Performance Benchmarks

**Typical Results:**
- Accuracy: 95-99%
- Precision: 0.95-0.99
- Recall: 0.95-0.99
- F1-Score: 0.95-0.99
- Inference Time: 0.1-1 ms per sample
- Model Size: 1-5 MB

**API Performance:**
- Single prediction: ~2-5 ms
- Batch (100 samples): ~10-20 ms
- Throughput: 200-500 predictions/second

---

## 📝 File Structure

```
.
├── ddos_detection.py      # Main training script
├── inference.py           # Inference engine
├── api.py                 # Flask REST API
├── config.py              # Configuration
├── test_setup.py          # Setup verification
├── test_api.py            # API testing
├── requirements.txt       # Dependencies
├── README.md              # Quick reference
├── QUICKSTART.md          # Quick start guide
├── DEPLOYMENT.md          # Deployment guide
└── COMPLETE_GUIDE.md      # This file
```

---

## 🎯 Next Steps

1. ✅ Install dependencies
2. ✅ Verify setup with `test_setup.py`
3. ✅ Train model with `ddos_detection.py`
4. ✅ Test inference with `inference.py`
5. ✅ Deploy API with `api.py`
6. ✅ Test API with `test_api.py`
7. ✅ Integrate into your application

---

## 📞 Support

For issues or questions:
1. Check this guide
2. Review error messages carefully
3. Verify all file paths are correct
4. Ensure all dependencies are installed
5. Check dataset format and labels

---

## 🔄 Model Updates

To retrain with new data:
1. Add new dataset to `config.py`
2. Run `python ddos_detection.py`
3. Model will be updated automatically
4. Restart API server to use new model

---

**Good luck with your DDoS detection system! 🚀**
