"""Quick verification script"""
import pickle
import os

print("="*60)
print("MODEL VERIFICATION")
print("="*60)

# 1. XGBoost
print("\n1. XGBoost Fine-tuned Model:")
try:
    with open('xgboost_waf_unified.pkl', 'rb') as f:
        xgb_data = pickle.load(f)
    print("   ✅ STATUS: WORKING")
    print(f"   - Features: {len(xgb_data['feature_names'])}")
    print(f"   - Datasets trained: {len(xgb_data['training_history'])}")
    for h in xgb_data['training_history']:
        print(f"     • {h['dataset']}: {h['results']['accuracy']:.2%} accuracy")
except Exception as e:
    print(f"   ❌ ERROR: {e}")

# 2. CNN-BiLSTM
print("\n2. CNN-BiLSTM Fine-tuned Model:")
try:
    import tensorflow as tf
    from tensorflow import keras
    from cnn_bilstm_waf_ldap import AttentionLayer
    
    model = keras.models.load_model(
        'cnn_bilstm_waf_model.h5',
        custom_objects={'AttentionLayer': AttentionLayer}
    )
    
    with open('cnn_bilstm_tokenizer.pkl', 'rb') as f:
        tokenizer_data = pickle.load(f)
    
    print("   ✅ STATUS: WORKING")
    print(f"   - Model parameters: {model.count_params():,}")
    print(f"   - Vocabulary size: {len(tokenizer_data['tokenizer'].word_index)}")
    print(f"   - Max sequence length: {tokenizer_data['max_length']}")
except Exception as e:
    print(f"   ❌ ERROR: {e}")

# 3. Ensemble
print("\n3. Ensemble Implementation:")
try:
    from ensemble_waf import EnsembleWAF
    
    waf = EnsembleWAF(strategy='cascading')
    waf.load_models()
    
    print("   ✅ STATUS: WORKING")
    print(f"   - Strategy: {waf.strategy}")
    print(f"   - XGBoost loaded: {waf.xgboost_model is not None}")
    print(f"   - CNN-BiLSTM loaded: {waf.cnn_bilstm_model is not None}")
    
    # Quick test
    test_request = {
        'url': '/test',
        'method': 'GET',
        'type': 'http'
    }
    result = waf.predict(test_request)
    print(f"   - Test prediction: {'Attack' if result['is_attack'] else 'Benign'}")
    print(f"   - Confidence: {result['confidence']:.2%}")
    print(f"   - Inference time: {result['inference_time_ms']:.2f}ms")
    
except Exception as e:
    print(f"   ❌ ERROR: {e}")

print("\n" + "="*60)
print("VERIFICATION COMPLETE")
print("="*60)
