"""
Installation Verification Script
Checks if all models and dependencies are properly installed
"""

import sys
import os

print("="*80)
print("WAF Privilege Escalation Detection - Installation Verification")
print("="*80)

# Check Python version
print(f"\n[1/5] Checking Python version...")
python_version = sys.version_info
print(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
    print("⚠ Warning: Python 3.8+ recommended")

# Check dependencies
print(f"\n[2/5] Checking dependencies...")
dependencies = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'sklearn': 'scikit-learn',
    'catboost': 'catboost',
    'lightgbm': 'lightgbm'
}

missing_deps = []
for module, package in dependencies.items():
    try:
        __import__(module)
        print(f"✓ {package} installed")
    except ImportError:
        print(f"✗ {package} NOT installed")
        missing_deps.append(package)

if missing_deps:
    print(f"\n⚠ Missing dependencies: {', '.join(missing_deps)}")
    print(f"Install with: pip install {' '.join(missing_deps)}")
else:
    print("\n✓ All dependencies installed")

# Check datasets
print(f"\n[3/5] Checking datasets...")
datasets = [
    'Attack_Dataset.csv',
    'CLOUD_VULRABILITES_DATASET.jsonl',
    'embedded_system_network_security_dataset.csv'
]

missing_datasets = []
for dataset in datasets:
    if os.path.exists(dataset):
        size_mb = os.path.getsize(dataset) / (1024 * 1024)
        print(f"✓ {dataset} ({size_mb:.2f} MB)")
    else:
        print(f"✗ {dataset} NOT found")
        missing_datasets.append(dataset)

if missing_datasets:
    print(f"\n⚠ Missing datasets: {', '.join(missing_datasets)}")
else:
    print("\n✓ All datasets present")

# Check model directory
print(f"\n[4/5] Checking model directory...")
if os.path.exists('model'):
    print("✓ model/ directory exists")
    
    model_files = [
        'catboost_waf_model.cbm',
        'catboost_waf_model.pkl',
        'lightgbm_waf_model.pkl',
        'lightgbm_waf_model.txt',
        'label_encoders.pkl',
        'feature_info.pkl'
    ]
    
    existing_models = []
    missing_models = []
    
    for model_file in model_files:
        model_path = os.path.join('model', model_file)
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"  ✓ {model_file} ({size_mb:.2f} MB)")
            existing_models.append(model_file)
        else:
            print(f"  ✗ {model_file} NOT found")
            missing_models.append(model_file)
    
    if missing_models:
        print(f"\n⚠ Missing model files: {len(missing_models)}/{len(model_files)}")
        print("Run: python waf_privilege_escalation_detection.py")
    else:
        print(f"\n✓ All model files present ({len(existing_models)}/{len(model_files)})")
else:
    print("✗ model/ directory NOT found")
    print("Run: python waf_privilege_escalation_detection.py")

# Check scripts
print(f"\n[5/5] Checking scripts...")
scripts = [
    'waf_privilege_escalation_detection.py',
    'load_and_predict.py',
    'model_evaluation_report.py'
]

for script in scripts:
    if os.path.exists(script):
        print(f"✓ {script}")
    else:
        print(f"✗ {script} NOT found")

# Final summary
print("\n" + "="*80)
print("VERIFICATION SUMMARY")
print("="*80)

all_good = (
    not missing_deps and 
    not missing_datasets and 
    os.path.exists('model') and
    all(os.path.exists(os.path.join('model', f)) for f in model_files)
)

if all_good:
    print("\n✅ ALL CHECKS PASSED!")
    print("\nYou're ready to use the models:")
    print("  1. Test predictions: python load_and_predict.py")
    print("  2. Generate report: python model_evaluation_report.py")
else:
    print("\n⚠ SOME CHECKS FAILED")
    
    if missing_deps:
        print(f"\n1. Install dependencies:")
        print(f"   pip install {' '.join(missing_deps)}")
    
    if missing_datasets:
        print(f"\n2. Add missing datasets to project directory")
    
    if not os.path.exists('model') or missing_models:
        print(f"\n3. Train models:")
        print(f"   python waf_privilege_escalation_detection.py")

print("\n" + "="*80)
print("For help, see README.md or QUICK_START.md")
print("="*80 + "\n")
