"""
Script to check and display results from trained models
"""

import os
import pickle
import glob

print("="*60)
print("LDAP Injection Detection WAF - Results Summary")
print("="*60)

# Check XGBoost model
print("\n" + "="*60)
print("XGBoost Model")
print("="*60)

if os.path.exists('xgboost_waf_model.pkl'):
    with open('xgboost_waf_model.pkl', 'rb') as f:
        xgb_data = pickle.load(f)
    
    print(f"\nModel file size: {os.path.getsize('xgboost_waf_model.pkl') / 1024:.2f} KB")
    print(f"Number of features: {len(xgb_data['feature_names'])}")
    print(f"\nTraining History:")
    
    for history in xgb_data['training_history']:
        print(f"\n  Dataset: {history['dataset']}")
        print(f"  Timestamp: {history['timestamp']}")
        results = history['results']
        print(f"    Accuracy:  {results['accuracy']:.4f}")
        print(f"    Precision: {results['precision']:.4f}")
        print(f"    Recall:    {results['recall']:.4f}")
        print(f"    F1-Score:  {results['f1_score']:.4f}")
        print(f"    ROC-AUC:   {results['roc_auc']:.4f}")
        print(f"    FPR:       {results['fpr']:.4f}")
else:
    print("\nXGBoost model not found!")

# Check CNN-BiLSTM model
print("\n\n" + "="*60)
print("CNN-BiLSTM Model")
print("="*60)

if os.path.exists('cnn_bilstm_waf_model.h5'):
    print(f"\nModel file size: {os.path.getsize('cnn_bilstm_waf_model.h5') / 1024:.2f} KB")
    
    if os.path.exists('cnn_bilstm_tokenizer.pkl'):
        with open('cnn_bilstm_tokenizer.pkl', 'rb') as f:
            cnn_data = pickle.load(f)
        
        print(f"Tokenizer file size: {os.path.getsize('cnn_bilstm_tokenizer.pkl') / 1024:.2f} KB")
        print(f"Max sequence length: {cnn_data['max_length']}")
        print(f"Vocabulary size: {len(cnn_data['tokenizer'].word_index)}")
        print(f"\nTraining History:")
        
        for history in cnn_data['training_history']:
            print(f"\n  Dataset: {history['dataset']}")
            print(f"  Timestamp: {history['timestamp']}")
            print(f"  Epochs trained: {history['epochs_trained']}")
            results = history['results']
            print(f"    Accuracy:  {results['accuracy']:.4f}")
            print(f"    Precision: {results['precision']:.4f}")
            print(f"    Recall:    {results['recall']:.4f}")
            print(f"    F1-Score:  {results['f1_score']:.4f}")
            print(f"    FPR:       {results['fpr']:.4f}")
else:
    print("\nCNN-BiLSTM model not found or still training!")

# Check generated visualizations
print("\n\n" + "="*60)
print("Generated Visualizations")
print("="*60)

viz_files = glob.glob('*.png')
if viz_files:
    print(f"\nFound {len(viz_files)} visualization files:")
    for f in sorted(viz_files):
        size = os.path.getsize(f) / 1024
        print(f"  - {f} ({size:.2f} KB)")
else:
    print("\nNo visualization files found!")

# Check best model checkpoints
print("\n\n" + "="*60)
print("Model Checkpoints")
print("="*60)

checkpoint_files = glob.glob('best_model_*.h5')
if checkpoint_files:
    print(f"\nFound {len(checkpoint_files)} checkpoint files:")
    for f in sorted(checkpoint_files):
        size = os.path.getsize(f) / 1024
        print(f"  - {f} ({size:.2f} KB)")
else:
    print("\nNo checkpoint files found!")

print("\n" + "="*60)
print("Summary Complete!")
print("="*60)
