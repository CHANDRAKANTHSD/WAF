"""
Test script to verify all dependencies are installed correctly
"""

import sys

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    print("-" * 60)
    
    packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'imblearn': 'imbalanced-learn',
        'lightgbm': 'lightgbm',
        'xgboost': 'xgboost',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'joblib': 'joblib'
    }
    
    failed = []
    
    for module, package in packages.items():
        try:
            __import__(module)
            print(f"✓ {package:25s} - OK")
        except ImportError as e:
            print(f"✗ {package:25s} - FAILED")
            failed.append(package)
    
    print("-" * 60)
    
    if failed:
        print(f"\n❌ {len(failed)} package(s) failed to import:")
        for pkg in failed:
            print(f"   - {pkg}")
        print("\nInstall missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True

def test_config():
    """Test if config file is accessible"""
    print("\nTesting configuration...")
    print("-" * 60)
    
    try:
        import config
        print(f"✓ Config file loaded")
        print(f"✓ Found {len(config.DATASETS)} datasets configured")
        print(f"✓ Feature selection: {config.N_FEATURES} features")
        print(f"✓ Test size: {config.TEST_SIZE}")
        print("-" * 60)
        print("\n✅ Configuration is valid!")
        return True
    except Exception as e:
        print(f"✗ Config error: {e}")
        print("-" * 60)
        print("\n❌ Configuration failed!")
        return False

def test_file_access():
    """Test if dataset files are accessible"""
    print("\nTesting dataset file access...")
    print("-" * 60)
    
    try:
        import config
        import os
        
        accessible = []
        inaccessible = []
        
        for dataset in config.DATASETS:
            path = dataset['path']
            name = dataset['name']
            
            if os.path.exists(path):
                print(f"✓ {name:30s} - Found")
                accessible.append(name)
            else:
                print(f"✗ {name:30s} - Not found")
                inaccessible.append(name)
        
        print("-" * 60)
        
        if inaccessible:
            print(f"\n⚠️  {len(inaccessible)} dataset(s) not accessible:")
            for name in inaccessible:
                print(f"   - {name}")
            print("\nUpdate paths in config.py to match your file locations")
            return False
        else:
            print(f"\n✅ All {len(accessible)} datasets are accessible!")
            return True
            
    except Exception as e:
        print(f"✗ Error: {e}")
        print("-" * 60)
        print("\n❌ File access test failed!")
        return False

def main():
    print("="*60)
    print("DDoS Detection System - Setup Test")
    print("="*60)
    print()
    
    results = []
    
    # Test imports
    results.append(test_imports())
    
    # Test config
    results.append(test_config())
    
    # Test file access
    results.append(test_file_access())
    
    # Final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if all(results):
        print("✅ All tests passed! You're ready to train the model.")
        print("\nNext step: Run 'python ddos_detection.py'")
        return 0
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
