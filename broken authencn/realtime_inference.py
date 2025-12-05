import numpy as np
import pandas as pd
import xgboost as xgb
from tensorflow import keras
import joblib
import time
import json

class RealtimeWAFDetector:
    """Real-time WAF authentication attack detector"""
    
    def __init__(self, model_type='xgboost'):
        """
        Initialize real-time detector
        
        Args:
            model_type: 'xgboost' or 'lstm'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_names = None
        
    def load_model(self, model_path=None, scaler_path='scaler.pkl', encoders_path='encoders.pkl'):
        """Load trained model and preprocessors"""
        print(f"Loading {self.model_type.upper()} model...")
        
        if model_path is None:
            model_path = 'xgboost_model.json' if self.model_type == 'xgboost' else 'lstm_model.h5'
        
        if self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier()
            self.model.load_model(model_path)
        else:
            self.model = keras.models.load_model(model_path)
        
        self.scaler = joblib.load(scaler_path)
        self.label_encoders = joblib.load(encoders_path)
        
        print(f"✓ Model loaded successfully")
    
    def preprocess_input(self, data):
        """Preprocess input data for inference"""
        df = pd.DataFrame([data]) if isinstance(data, dict) else data.copy()
        
        # Encode categorical features
        if 'device_type' in df.columns:
            df['device_type'] = self.label_encoders['device_type'].transform(df['device_type'].astype(str))
        
        # Scale features
        X = self.scaler.transform(df)
        
        return X
    
    def predict(self, data, return_proba=True):
        """
        Make real-time prediction
        
        Args:
            data: dict or DataFrame with features
            return_proba: return probability scores
            
        Returns:
            prediction, probability, latency_ms
        """
        start_time = time.time()
        
        # Preprocess
        X = self.preprocess_input(data)
        
        # Predict
        if self.model_type == 'xgboost':
            prediction = self.model.predict(X)[0]
            if return_proba:
                probability = self.model.predict_proba(X)[0][1]
            else:
                probability = None
        else:
            X_lstm = X.reshape((X.shape[0], 1, X.shape[1]))
            probability = self.model.predict(X_lstm, verbose=0)[0][0]
            prediction = 1 if probability > 0.5 else 0
        
        latency_ms = (time.time() - start_time) * 1000
        
        return {
            'prediction': int(prediction),
            'probability': float(probability) if probability is not None else None,
            'latency_ms': latency_ms,
            'is_attack': bool(prediction),
            'risk_level': self._get_risk_level(probability if probability is not None else prediction)
        }
    
    def _get_risk_level(self, score):
        """Determine risk level from score"""
        if score < 0.3:
            return 'LOW'
        elif score < 0.7:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def batch_predict(self, data_list):
        """Batch prediction for multiple samples"""
        results = []
        total_start = time.time()
        
        for data in data_list:
            result = self.predict(data)
            results.append(result)
        
        total_time = (time.time() - total_start) * 1000
        avg_latency = total_time / len(data_list)
        
        return {
            'results': results,
            'total_samples': len(data_list),
            'total_time_ms': total_time,
            'avg_latency_ms': avg_latency
        }


def demo_realtime_inference():
    """Demonstrate real-time inference"""
    print("="*70)
    print("REAL-TIME WAF AUTHENTICATION DETECTION DEMO")
    print("="*70)
    
    # Test samples with all features
    test_samples = [
        {
            'login_attempts': 5,
            'failed_attempts': 0,
            'failed_ratio': 0.0,
            'session_duration': 1200,
            'ip_changes': 1,
            'country_changes': 0,
            'abnormal_rtt': 0,
            'device_type': 'mobile',
            'hour': 14,
            'day_of_week': 2,
            'is_night': 0,
            'is_weekend': 0
        },
        {
            'login_attempts': 50,
            'failed_attempts': 30,
            'failed_ratio': 0.6,
            'session_duration': 120,
            'ip_changes': 8,
            'country_changes': 5,
            'abnormal_rtt': 1,
            'device_type': 'desktop',
            'hour': 3,
            'day_of_week': 6,
            'is_night': 1,
            'is_weekend': 1
        },
        {
            'login_attempts': 100,
            'failed_attempts': 80,
            'failed_ratio': 0.8,
            'session_duration': 30,
            'ip_changes': 15,
            'country_changes': 10,
            'abnormal_rtt': 1,
            'device_type': 'unknown',
            'hour': 2,
            'day_of_week': 0,
            'is_night': 1,
            'is_weekend': 0
        }
    ]
    
    # Test both models
    for model_type in ['xgboost', 'lstm']:
        print(f"\n{'='*70}")
        print(f"Testing {model_type.upper()} Model")
        print(f"{'='*70}")
        
        detector = RealtimeWAFDetector(model_type=model_type)
        
        try:
            detector.load_model()
            
            print(f"\nSingle Prediction Tests:")
            print("-" * 70)
            
            for i, sample in enumerate(test_samples, 1):
                result = detector.predict(sample)
                print(f"\nSample {i}:")
                print(f"  Login Attempts: {sample['login_attempts']}, Failed: {sample['failed_attempts']}")
                print(f"  Prediction: {'ATTACK' if result['is_attack'] else 'BENIGN'}")
                print(f"  Risk Level: {result['risk_level']}")
                print(f"  Probability: {result['probability']:.4f}")
                print(f"  Latency: {result['latency_ms']:.2f} ms {'✓' if result['latency_ms'] < 100 else '✗'}")
            
            # Batch prediction
            print(f"\n{'='*70}")
            print(f"Batch Prediction Test (1000 samples)")
            print(f"{'='*70}")
            
            batch_samples = [test_samples[0]] * 1000
            batch_results = detector.batch_predict(batch_samples)
            
            print(f"\nTotal Samples: {batch_results['total_samples']}")
            print(f"Total Time: {batch_results['total_time_ms']:.2f} ms")
            print(f"Average Latency: {batch_results['avg_latency_ms']:.2f} ms per sample")
            print(f"Throughput: {1000 / batch_results['avg_latency_ms']:.0f} predictions/second")
            
            status = "✓ PASS" if batch_results['avg_latency_ms'] < 100 else "✗ FAIL"
            print(f"Real-time Requirement (<100ms): {status}")
            
        except Exception as e:
            print(f"Error loading {model_type} model: {e}")
            print("Please run waf_model.py first to train the models.")

if __name__ == "__main__":
    demo_realtime_inference()
