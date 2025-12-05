import joblib
import pickle
import pandas as pd
import numpy as np
import time

class DDoSInference:
    def __init__(self, model_path, scaler_path, features_path):
        """Initialize inference engine"""
        print("Loading model and preprocessing components...")
        
        # Load model
        self.model = joblib.load(model_path)
        print(f"✓ Model loaded from {model_path}")
        
        # Load scaler
        self.scaler = joblib.load(scaler_path)
        print(f"✓ Scaler loaded from {scaler_path}")
        
        # Load selected features
        with open(features_path, 'rb') as f:
            self.selected_features = pickle.load(f)
        print(f"✓ Selected features loaded ({len(self.selected_features)} features)")
        
    def preprocess(self, X):
        """Preprocess input data"""
        # Ensure all selected features are present
        for feature in self.selected_features:
            if feature not in X.columns:
                X[feature] = 0
        
        # Select only the features used during training
        X_selected = X[self.selected_features]
        
        # Handle non-numeric values
        for col in X_selected.columns:
            if X_selected[col].dtype == 'object':
                X_selected[col] = pd.to_numeric(X_selected[col], errors='coerce')
        
        # Handle infinite and missing values
        X_selected.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_selected.fillna(0, inplace=True)
        
        # Scale features
        X_scaled = self.scaler.transform(X_selected.values if hasattr(X_selected, 'values') else X_selected)
        
        return X_scaled
    
    def predict(self, X):
        """Make predictions on input data"""
        start_time = time.time()
        
        # Preprocess
        X_processed = self.preprocess(X)
        
        # Predict
        predictions = self.model.predict(X_processed)
        probabilities = self.model.predict_proba(X_processed)
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        return predictions, probabilities, inference_time
    
    def predict_single(self, sample):
        """Predict single sample"""
        if isinstance(sample, dict):
            sample = pd.DataFrame([sample])
        elif isinstance(sample, pd.Series):
            sample = sample.to_frame().T
        
        predictions, probabilities, inference_time = self.predict(sample)
        
        result = {
            'prediction': 'DDoS Attack' if predictions[0] == 1 else 'Normal',
            'confidence': float(probabilities[0][predictions[0]]),
            'ddos_probability': float(probabilities[0][1]),
            'normal_probability': float(probabilities[0][0]),
            'inference_time_ms': inference_time
        }
        
        return result

# Example usage
if __name__ == "__main__":
    # Initialize inference engine
    # XGBoost was selected as the best model
    inference_engine = DDoSInference(
        model_path='XGBoost_ddos_model.joblib',
        scaler_path='scaler.joblib',
        features_path='selected_features.pkl'
    )
    
    print("\n" + "="*80)
    print("DDoS Detection Inference Engine Ready")
    print("="*80)
    
    # Example: Load test data
    print("\nExample: Loading test data...")
    try:
        test_data = pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 
                                encoding='latin1', nrows=10)
        
        # Remove label column if present
        label_cols = [' Label', 'Label', 'label', 'type', 'Type']
        for col in label_cols:
            if col in test_data.columns:
                test_data = test_data.drop(columns=[col])
        
        # Make predictions
        predictions, probabilities, inference_time = inference_engine.predict(test_data)
        
        print(f"\nPredictions: {predictions}")
        print(f"Inference time: {inference_time:.2f} ms for {len(test_data)} samples")
        print(f"Average: {inference_time/len(test_data):.4f} ms/sample")
        
        # Show detailed results
        print("\nDetailed Results:")
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            status = "DDoS Attack" if pred == 1 else "Normal"
            confidence = prob[pred] * 100
            print(f"Sample {i+1}: {status} (Confidence: {confidence:.2f}%)")
            
    except FileNotFoundError:
        print("Test data file not found. Use your own data for testing.")
        print("\nExample single prediction:")
        
        # Create dummy sample
        sample = {feature: 0.0 for feature in inference_engine.selected_features}
        result = inference_engine.predict_single(sample)
        
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        print(f"DDoS Probability: {result['ddos_probability']*100:.2f}%")
        print(f"Inference Time: {result['inference_time_ms']:.4f} ms")
