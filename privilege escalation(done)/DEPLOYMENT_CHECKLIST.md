# 📦 Complete Deployment Checklist

## ✅ Files Required for Production Deployment

### 🎯 MINIMUM FILES (Required)

Copy these 5 files to your production server:

```
your_backend_project/
├── model/
│   ├── catboost_waf_model.cbm          ✅ REQUIRED (3-5 MB)
│   ├── lightgbm_waf_model.pkl          ✅ REQUIRED (2-3 MB)
│   ├── label_encoders.pkl              ✅ REQUIRED (~50 KB)
│   └── feature_info.pkl                ✅ REQUIRED (~10 KB)
│
└── ensemble_model.py                    ✅ REQUIRED (~10 KB)
```

**Total Size**: ~5-10 MB

---

## 📋 Detailed File List

### 1. Model Files (4 files in `model/` folder)

#### ✅ `model/catboost_waf_model.cbm`
- **What**: Trained CatBoost model
- **Size**: 3-5 MB
- **Purpose**: Makes predictions with CatBoost algorithm
- **Location**: `D:\Major_project\Honnushree\privilege escalation\model\catboost_waf_model.cbm`

#### ✅ `model/lightgbm_waf_model.pkl`
- **What**: Trained LightGBM model
- **Size**: 2-3 MB
- **Purpose**: Makes predictions with LightGBM algorithm
- **Location**: `D:\Major_project\Honnushree\privilege escalation\model\lightgbm_waf_model.pkl`

#### ✅ `model/label_encoders.pkl`
- **What**: Encoders for categorical features
- **Size**: ~50 KB
- **Purpose**: Converts text features (like "AWS", "Azure") to numbers
- **Location**: `D:\Major_project\Honnushree\privilege escalation\model\label_encoders.pkl`

#### ✅ `model/feature_info.pkl`
- **What**: Feature metadata
- **Size**: ~10 KB
- **Purpose**: Stores which features are categorical vs numerical
- **Location**: `D:\Major_project\Honnushree\privilege escalation\model\feature_info.pkl`

---

### 2. Ensemble Wrapper (1 file)

#### ✅ `ensemble_model.py`
- **What**: Python class that loads both models and combines predictions
- **Size**: ~10 KB
- **Purpose**: Provides easy-to-use API for ensemble predictions
- **Location**: `D:\Major_project\Honnushree\privilege escalation\ensemble_model.py`

---

## 📦 Optional Files (Not Required for Deployment)

These files are useful for reference but NOT needed in production:

### Documentation Files (Optional)
```
❌ NOT NEEDED for deployment:
├── MODEL_COMPARISON_GUIDE.md       (Reference only)
├── DEPLOYMENT_GUIDE.md             (Reference only)
├── ENSEMBLE_SUMMARY.md             (Reference only)
├── DEPLOYMENT_CHECKLIST.md         (This file - reference only)
├── README.md                       (Reference only)
└── QUICK_START.md                  (Reference only)
```

### Training/Testing Files (Optional)
```
❌ NOT NEEDED for deployment:
├── waf_privilege_escalation_detection.py    (Training script)
├── load_and_predict.py                      (Testing script)
├── verify_installation.py                   (Testing script)
├── model_evaluation_report.py               (Evaluation script)
└── test_system.py                           (Testing script)
```

### Dataset Files (Optional)
```
❌ NOT NEEDED for deployment:
├── CLOUD_VULRABILITES_DATASET.jsonl         (Training data)
├── Attack_Dataset.csv                       (Training data)
└── embedded_system_network_security_dataset.csv  (Training data)
```

---

## 🚀 Step-by-Step Deployment

### Step 1: Create Deployment Package

On your local machine:

```bash
# Create deployment folder
mkdir waf_deployment
cd waf_deployment

# Copy model folder
cp -r "D:\Major_project\Honnushree\privilege escalation\model" .

# Copy ensemble wrapper
cp "D:\Major_project\Honnushree\privilege escalation\ensemble_model.py" .

# Your deployment package is ready!
```

**Result**:
```
waf_deployment/
├── model/
│   ├── catboost_waf_model.cbm
│   ├── lightgbm_waf_model.pkl
│   ├── label_encoders.pkl
│   └── feature_info.pkl
└── ensemble_model.py
```

---

### Step 2: Copy to Production Server

#### Option A: Using SCP (Linux/Mac)
```bash
scp -r waf_deployment/ user@your-server:/path/to/backend/
```

#### Option B: Using FTP/SFTP
1. Connect to your server via FTP client (FileZilla, WinSCP)
2. Upload the `waf_deployment` folder
3. Place it in your backend project directory

#### Option C: Using Git
```bash
# Add to .gitignore (if files are large)
echo "*.cbm" >> .gitignore
echo "*.pkl" >> .gitignore

# Or commit if files are small enough
git add model/ ensemble_model.py
git commit -m "Add trained models for deployment"
git push

# On server
git pull
```

#### Option D: Manual Copy (Windows)
1. Copy the `waf_deployment` folder
2. Paste it into your backend project on the server

---

### Step 3: Install Dependencies on Server

```bash
# SSH into your server
ssh user@your-server

# Navigate to your project
cd /path/to/your/backend

# Install required packages
pip install catboost lightgbm pandas numpy scikit-learn

# Or use requirements.txt
pip install -r requirements.txt
```

**requirements.txt**:
```
catboost==1.2.0
lightgbm==4.0.0
pandas==2.0.0
numpy==1.24.0
scikit-learn==1.3.0
```

---

### Step 4: Verify Deployment

Create a test script on your server:

**test_deployment.py**:
```python
from ensemble_model import EnsembleWAFDetector

print("Testing deployment...")

# Initialize detector
detector = EnsembleWAFDetector()

# Test prediction
features = {
    'attack_category': 'IAM Misconfiguration',
    'attack_type': 'Privilege Escalation',
    'target_system': 'AWS',
    'mitre_technique': 'T1078 (Valid Accounts)',
    'packet_size': 0.5,
    'inter_arrival_time': 0.3,
    'packet_count_5s': 0.8,
    'mean_packet_size': 0.0,
    'spectral_entropy': 0.7,
    'frequency_band_energy': 0.6
}

result = detector.predict(features)

print(f"✓ Prediction: {result['ensemble']['label']}")
print(f"✓ Confidence: {result['ensemble']['confidence_percent']:.2f}%")
print(f"✓ Risk Level: {result['ensemble']['risk_level']}")
print("\n✅ Deployment successful!")
```

Run it:
```bash
python test_deployment.py
```

**Expected Output**:
```
Loading Ensemble WAF Detector...
✓ CatBoost model loaded
✓ LightGBM model loaded
✓ Label encoders loaded
✓ Feature info loaded
✓ Ensemble detector ready!

Testing deployment...
✓ Prediction: PRIVILEGE_ESCALATION
✓ Confidence: 74.66%
✓ Risk Level: HIGH

✅ Deployment successful!
```

---

## 🔧 Integration with Your Backend

### Flask Example

**app.py**:
```python
from flask import Flask, request, jsonify
from ensemble_model import EnsembleWAFDetector

app = Flask(__name__)

# Initialize detector once at startup
detector = EnsembleWAFDetector()

@app.route('/api/check-request', methods=['POST'])
def check_request():
    data = request.json
    
    features = {
        'attack_category': data.get('category'),
        'attack_type': data.get('type'),
        'target_system': data.get('system'),
        'mitre_technique': data.get('technique'),
        'packet_size': float(data.get('packet_size', 0)),
        'inter_arrival_time': float(data.get('inter_arrival_time', 0)),
        'packet_count_5s': float(data.get('packet_count', 0)),
        'mean_packet_size': float(data.get('mean_packet_size', 0)),
        'spectral_entropy': float(data.get('entropy', 0)),
        'frequency_band_energy': float(data.get('energy', 0))
    }
    
    result = detector.predict(features)
    
    return jsonify({
        'prediction': result['ensemble']['label'],
        'confidence': result['ensemble']['confidence_percent'],
        'risk_level': result['ensemble']['risk_level'],
        'probability': result['ensemble']['probability']
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Your project structure**:
```
your_backend/
├── model/                          ← Copied from deployment package
│   ├── catboost_waf_model.cbm
│   ├── lightgbm_waf_model.pkl
│   ├── label_encoders.pkl
│   └── feature_info.pkl
├── ensemble_model.py               ← Copied from deployment package
├── app.py                          ← Your Flask app
└── requirements.txt
```

---

## 📊 File Size Summary

| File | Size | Required? |
|------|------|-----------|
| `model/catboost_waf_model.cbm` | 3-5 MB | ✅ YES |
| `model/lightgbm_waf_model.pkl` | 2-3 MB | ✅ YES |
| `model/label_encoders.pkl` | ~50 KB | ✅ YES |
| `model/feature_info.pkl` | ~10 KB | ✅ YES |
| `ensemble_model.py` | ~10 KB | ✅ YES |
| **TOTAL** | **~5-10 MB** | **5 files** |

---

## ✅ Pre-Deployment Checklist

Before deploying, verify:

- [ ] All 5 files are present
- [ ] Model files are not corrupted (check file sizes)
- [ ] `ensemble_model.py` is the latest version
- [ ] Dependencies are listed in `requirements.txt`
- [ ] Test script runs successfully locally
- [ ] Server has Python 3.7+ installed
- [ ] Server has sufficient RAM (minimum 2GB)
- [ ] Server has sufficient disk space (minimum 100MB)

---

## 🐳 Docker Deployment (Optional)

If using Docker:

**Dockerfile**:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files
COPY model/ ./model/
COPY ensemble_model.py .
COPY app.py .

EXPOSE 5000

CMD ["python", "app.py"]
```

**Build and run**:
```bash
docker build -t waf-detector .
docker run -p 5000:5000 waf-detector
```

---

## 🔒 Security Considerations

### File Permissions
```bash
# Set appropriate permissions
chmod 644 model/*.cbm
chmod 644 model/*.pkl
chmod 644 ensemble_model.py
```

### Environment Variables
```bash
# Don't hardcode paths, use environment variables
export MODEL_DIR=/path/to/model
```

**In code**:
```python
import os
model_dir = os.getenv('MODEL_DIR', 'model')
detector = EnsembleWAFDetector(model_dir=model_dir)
```

---

## 📝 Quick Copy Commands

### Copy All Required Files (Windows PowerShell)
```powershell
# Create deployment folder
New-Item -ItemType Directory -Path "waf_deployment"

# Copy model folder
Copy-Item -Path "D:\Major_project\Honnushree\privilege escalation\model" -Destination "waf_deployment\" -Recurse

# Copy ensemble wrapper
Copy-Item -Path "D:\Major_project\Honnushree\privilege escalation\ensemble_model.py" -Destination "waf_deployment\"

Write-Host "✅ Deployment package ready in waf_deployment folder"
```

### Copy All Required Files (Linux/Mac)
```bash
# Create deployment folder
mkdir -p waf_deployment

# Copy model folder
cp -r "model" waf_deployment/

# Copy ensemble wrapper
cp "ensemble_model.py" waf_deployment/

echo "✅ Deployment package ready in waf_deployment folder"
```

---

## 🎯 Final Deployment Structure

Your production server should have:

```
/var/www/your_backend/          (or wherever your backend is)
├── model/
│   ├── catboost_waf_model.cbm          ✅
│   ├── lightgbm_waf_model.pkl          ✅
│   ├── label_encoders.pkl              ✅
│   └── feature_info.pkl                ✅
├── ensemble_model.py                    ✅
├── app.py                               (your Flask/FastAPI app)
├── requirements.txt
└── ... (other backend files)
```

---

## 🚀 Quick Start Commands

### On Your Local Machine:
```bash
# 1. Create deployment package
mkdir waf_deployment
cp -r model waf_deployment/
cp ensemble_model.py waf_deployment/
```

### On Your Server:
```bash
# 2. Install dependencies
pip install catboost lightgbm pandas numpy scikit-learn

# 3. Test deployment
python -c "from ensemble_model import EnsembleWAFDetector; d = EnsembleWAFDetector(); print('✅ Ready!')"
```

---

## 📞 Troubleshooting

### Error: "No module named 'catboost'"
```bash
pip install catboost
```

### Error: "No module named 'lightgbm'"
```bash
pip install lightgbm
```

### Error: "FileNotFoundError: model/catboost_waf_model.cbm"
- Check that `model/` folder is in the same directory as `ensemble_model.py`
- Verify all 4 files are in the `model/` folder

### Error: "Model file is corrupted"
- Re-copy the model files from your local machine
- Verify file sizes match the original

---

## ✨ Summary

### Required Files (5 total):
1. ✅ `model/catboost_waf_model.cbm`
2. ✅ `model/lightgbm_waf_model.pkl`
3. ✅ `model/label_encoders.pkl`
4. ✅ `model/feature_info.pkl`
5. ✅ `ensemble_model.py`

### Total Size: ~5-10 MB

### Dependencies:
```bash
pip install catboost lightgbm pandas numpy scikit-learn
```

### Usage:
```python
from ensemble_model import EnsembleWAFDetector
detector = EnsembleWAFDetector()
result = detector.predict(features)
```

**You're ready to deploy!** 🎉
