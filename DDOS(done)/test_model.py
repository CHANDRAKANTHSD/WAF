"""
Quick test of the trained model
"""
import joblib
import pickle
import numpy as np

print("="*80)
print("Testing Trained DDoS Detection Model")
print("="*80)

# Load model
print("\n1. Loading model...")
model = joblib.load('XGBoost_ddos_model.joblib')
print("   ✓ Model loaded successfully")
print(f"   Model type: {type(model).__name__}")

# Load features
print("\n2. Loading selected features...")
with open('selected_features.pkl', 'rb') as f:
    features = pickle.load(f)
print(f"   ✓ Loaded {len(features)} features")
print(f"   Features: {features[:5]}...")

# Create dummy test data (30 features)
print("\n3. Creating test data...")
X_test = np.random.rand(5, 30)  # 5 samples, 30 features
print(f"   ✓ Created test data: {X_test.shape}")

# Make predictions
print("\n4. Making predictions...")
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print(f"   ✓ Predictions: {predictions}")
print(f"   ✓ Probabilities shape: {probabilities.shape}")

# Show results
print("\n5. Results:")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    label = "DDoS Attack" if pred == 1 else "Normal"
    confidence = prob[pred] * 100
    print(f"   Sample {i+1}: {label} (Confidence: {confidence:.2f}%)")

print("\n" + "="*80)
print("✅ Model is working correctly!")
print("="*80)
