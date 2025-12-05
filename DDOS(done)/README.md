# DDoS Detection System - LightGBM vs XGBoost

A comprehensive DDoS detection system for Web Application Firewalls (WAF) that trains and compares LightGBM and XGBoost models on multiple datasets, automatically selecting the best performer.

## 🚀 Quick Start (Windows)

```bash
# 1. Install dependencies
install_dependencies.bat

# 2. Train the model
run_training.bat

# 3. Start API server
run_api.bat
```

## 🎯 Features

- **Multi-Dataset Training**: Sequential training on 7 datasets (CICIDS2017, CSE-CIC-IDS2018, TON_IoT)
- **Automatic Model Selection**: Compares LightGBM vs XGBoost, selects best based on F1-Score
- **Feature Engineering**: Optimal feature selection (top 30 features)
- **Class Imbalance Handling**: SMOTE oversampling for balanced training
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, Inference Time
- **Feature Importance**: Visualization of most important features
- **REST API**: Flask API for easy integration
- **Multiple Export Formats**: .pkl and .joblib for flexibility
- **Production Ready**: Includes monitoring, testing, and deployment guides

## 📦 Installation

### Option 1: Using Batch Script (Windows)
```bash
install_dependencies.bat
```

### Option 2: Manual Installation
```bash
pip install -r requirements.txt
```

### Verify Installation
```bash
python test_setup.py
```

## 🎓 Training the Model

### Quick Training
```bash
run_training.bat
```

### Manual Training
```bash
python ddos_detection.py
```

**Training Process:**
1. Loads 7 datasets sequentially
2. Preprocesses and cleans data
3. Selects optimal features
4. Handles class imbalance
5. Trains both LightGBM and XGBoost
6. Compares performance
7. Saves best model

**Duration:** 10-30 minutes (hardware dependent)

## 🔮 Using the Model

### Option 1: REST API (Recommended)

Start the server:
```bash
run_api.bat
# or
python api.py
```

Make predictions:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"flow_duration": 1000, "total_fwd_packets": 10}'
```

### Option 2: Python Integration

```python
from inference import DDoSInference

# Initialize
detector = DDoSInference(
    model_path='LightGBM_ddos_model.joblib',
    scaler_path='scaler.joblib',
    features_path='selected_features.pkl'
)

# Predict
result = detector.predict_single(network_features)
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']*100:.2f}%")
```

## 📊 Output Files

After training, you'll have:
- `LightGBM_ddos_model.pkl` or `XGBoost_ddos_model.pkl` - Model (pickle format)
- `LightGBM_ddos_model.joblib` or `XGBoost_ddos_model.joblib` - Model (joblib format)
- `scaler.joblib` - Feature scaler
- `selected_features.pkl` - Selected feature names
- `*_feature_importance.png` - Feature importance visualization

## 🌐 API Endpoints

- `GET /` - Service information
- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `GET /model/info` - Model information

## 📈 Performance Expectations

- **Accuracy**: 95-99%
- **F1-Score**: 0.95-0.99
- **Inference Time**: 0.1-1 ms per sample
- **API Response**: 2-5 ms per request
- **Model Size**: 1-5 MB

## 📚 Documentation

- **README.md** (this file) - Quick reference
- **QUICKSTART.md** - Step-by-step quick start guide
- **COMPLETE_GUIDE.md** - Comprehensive documentation
- **DEPLOYMENT.md** - Deployment and integration strategies
- **WORKFLOW_DIAGRAM.txt** - Visual workflow diagrams
- **PROJECT_SUMMARY.txt** - Project overview
- **CHECKLIST.md** - Implementation checklist

## 🗂️ Project Structure

```
├── ddos_detection.py          # Main training script
├── inference.py               # Inference engine
├── api.py                     # Flask REST API
├── config.py                  # Configuration
├── test_setup.py              # Setup verification
├── test_api.py                # API testing
├── requirements.txt           # Dependencies
├── install_dependencies.bat   # Windows installer
├── run_training.bat           # Windows training script
├── run_api.bat                # Windows API launcher
└── Documentation files
```

## 🔧 Configuration

Edit `config.py` to customize:
- Dataset paths
- Number of features (default: 30)
- Model hyperparameters
- Train/test split ratio

## 🧪 Testing

```bash
# Test setup
python test_setup.py

# Test inference
python inference.py

# Test API
python test_api.py
```

## 📊 Datasets Used

1. **CICIDS2017** (3 files)
   - Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
   - Monday-WorkingHours.pcap_ISCX.csv
   - Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv

2. **CSE-CIC-IDS2018** (3 files)
   - DDOS attack-HOIC.csv
   - DDOS attack-LOIC-UDP.csv
   - DDoS attacks-LOIC-HTTP.csv

3. **TON_IoT** (1 file)
   - ton_iot_network.csv

## 🔒 Security Features

- Input validation
- Rate limiting support
- CORS enabled
- Error handling
- Logging

## 🐛 Troubleshooting

**Model not found?**
- Run training first: `run_training.bat`

**Dataset file not found?**
- Update paths in `config.py`

**Memory error?**
- Reduce `N_FEATURES` in `config.py`

**API connection refused?**
- Ensure server is running: `run_api.bat`

See **COMPLETE_GUIDE.md** for detailed troubleshooting.

## 📞 Support

For detailed help, refer to:
- **COMPLETE_GUIDE.md** - Comprehensive documentation
- **CHECKLIST.md** - Step-by-step checklist
- **WORKFLOW_DIAGRAM.txt** - Visual workflows

## 🎯 Next Steps

1. ✅ Install dependencies
2. ✅ Verify setup
3. ✅ Train model
4. ✅ Test predictions
5. ✅ Deploy API
6. ✅ Integrate with your WAF

## 📄 License

This project is provided as-is for educational and research purposes.

---

**Ready to detect DDoS attacks? Start with `install_dependencies.bat`! 🚀**
