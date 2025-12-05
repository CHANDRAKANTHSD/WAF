"""
Ensemble Model Wrapper for Full-Stack Deployment
Combines LightGBM and CatBoost models for optimal WAF privilege escalation detection

Usage:
    from ensemble_model import EnsembleWAFDetector
    
    detector = EnsembleWAFDetector()
    result = detector.predict(features)
"""

import pickle
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


class EnsembleWAFDetector:
    """
    Ensemble model combining LightGBM and CatBoost for WAF privilege escalation detection
    
    This class loads both trained models and provides ensemble predictions by averaging
    their probability outputs, giving you the best of both models.
    """
    
    def __init__(self, model_dir='model'):
        """
        Initialize the ensemble detector by loading both models
        
        Args:
            model_dir (str): Directory containing the model files
        """
        self.model_dir = model_dir
        self.catboost_model = None
        self.lightgbm_model = None
        self.label_encoders = None
        self.feature_info = None
        
        self._load_models()
    
    def _load_models(self):
        """Load all required models and encoders"""
        print("Loading Ensemble WAF Detector...")
        
        # Load CatBoost model
        self.catboost_model = CatBoostClassifier()
        self.catboost_model.load_model(f'{self.model_dir}/catboost_waf_model.cbm')
        print("✓ CatBoost model loaded")
        
        # Load LightGBM model
        with open(f'{self.model_dir}/lightgbm_waf_model.pkl', 'rb') as f:
            self.lightgbm_model = pickle.load(f)
        print("✓ LightGBM model loaded")
        
        # Load label encoders
        with open(f'{self.model_dir}/label_encoders.pkl', 'rb') as f:
            self.label_encoders = pickle.load(f)
        print("✓ Label encoders loaded")
        
        # Load feature info
        with open(f'{self.model_dir}/feature_info.pkl', 'rb') as f:
            self.feature_info = pickle.load(f)
        print("✓ Feature info loaded")
        
        print("✓ Ensemble detector ready!\n")
    
    def _prepare_features(self, features):
        """
        Prepare features for prediction
        
        Args:
            features (dict or pd.DataFrame): Input features
            
        Returns:
            tuple: (features_for_catboost, features_for_lightgbm)
        """
        # Convert dict to DataFrame if needed
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        else:
            features_df = features.copy()
        
        # For CatBoost (handles categorical features natively)
        features_cb = features_df.copy()
        
        # For LightGBM (needs encoded categorical features)
        features_lgb = features_df.copy()
        for col in self.feature_info['categorical_features']:
            if col in self.label_encoders and col in features_lgb.columns:
                features_lgb[col] = features_lgb[col].apply(
                    lambda x: self.label_encoders[col].transform([str(x)])[0] 
                    if str(x) in self.label_encoders[col].classes_ 
                    else 0
                )
        
        return features_cb, features_lgb
    
    def predict(self, features, threshold=0.5, return_details=False):
        """
        Make ensemble prediction
        
        Args:
            features (dict or pd.DataFrame): Input features
            threshold (float): Classification threshold (default: 0.5)
            return_details (bool): If True, return detailed predictions from both models
            
        Returns:
            dict: Prediction results with ensemble decision
        """
        # Prepare features
        features_cb, features_lgb = self._prepare_features(features)
        
        # CatBoost prediction
        cb_pred = self.catboost_model.predict(features_cb)[0]
        cb_proba = self.catboost_model.predict_proba(features_cb)[0][1]
        
        # LightGBM prediction
        lgb_pred = self.lightgbm_model.predict(features_lgb)[0]
        lgb_proba = self.lightgbm_model.predict_proba(features_lgb)[0][1]
        
        # Ensemble prediction (average probabilities)
        ensemble_proba = (cb_proba + lgb_proba) / 2
        ensemble_pred = 1 if ensemble_proba > threshold else 0
        
        # Prepare result
        result = {
            'ensemble': {
                'prediction': int(ensemble_pred),
                'probability': float(ensemble_proba),
                'label': 'PRIVILEGE_ESCALATION' if ensemble_pred == 1 else 'NORMAL',
                'confidence_percent': float(ensemble_proba * 100),
                'risk_level': self._get_risk_level(ensemble_proba)
            }
        }
        
        # Add detailed predictions if requested
        if return_details:
            result['catboost'] = {
                'prediction': int(cb_pred),
                'probability': float(cb_proba),
                'label': 'PRIVILEGE_ESCALATION' if cb_pred == 1 else 'NORMAL'
            }
            result['lightgbm'] = {
                'prediction': int(lgb_pred),
                'probability': float(lgb_proba),
                'label': 'PRIVILEGE_ESCALATION' if lgb_pred == 1 else 'NORMAL'
            }
        
        return result
    
    def predict_batch(self, features_list, threshold=0.5):
        """
        Make predictions for multiple samples
        
        Args:
            features_list (list): List of feature dictionaries
            threshold (float): Classification threshold
            
        Returns:
            list: List of prediction results
        """
        results = []
        for features in features_list:
            result = self.predict(features, threshold=threshold)
            results.append(result)
        return results
    
    def _get_risk_level(self, probability):
        """
        Determine risk level based on probability
        
        Args:
            probability (float): Ensemble probability
            
        Returns:
            str: Risk level (LOW, MEDIUM, HIGH, CRITICAL)
        """
        if probability >= 0.8:
            return 'CRITICAL'
        elif probability >= 0.6:
            return 'HIGH'
        elif probability >= 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def get_model_info(self):
        """
        Get information about the loaded models
        
        Returns:
            dict: Model information
        """
        return {
            'models': ['CatBoost', 'LightGBM'],
            'ensemble_method': 'Average Probability',
            'categorical_features': self.feature_info['categorical_features'],
            'numerical_features': self.feature_info['numerical_features'],
            'total_features': len(self.feature_info['categorical_features']) + 
                            len(self.feature_info['numerical_features'])
        }


# ============================================================================
# Example Usage for Full-Stack Deployment
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("ENSEMBLE WAF DETECTOR - DEPLOYMENT EXAMPLE")
    print("="*80 + "\n")
    
    # Initialize detector (do this once at application startup)
    detector = EnsembleWAFDetector()
    
    # Get model info
    info = detector.get_model_info()
    print(f"Loaded models: {', '.join(info['models'])}")
    print(f"Ensemble method: {info['ensemble_method']}")
    print(f"Total features: {info['total_features']}\n")
    
    # Example 1: Single prediction
    print("="*80)
    print("EXAMPLE 1: Single Prediction")
    print("="*80)
    
    sample_request = {
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
    
    result = detector.predict(sample_request, return_details=True)
    
    print(f"\nInput: {sample_request['attack_type']} on {sample_request['target_system']}")
    print(f"\n🎯 ENSEMBLE DECISION:")
    print(f"   Prediction: {result['ensemble']['label']}")
    print(f"   Confidence: {result['ensemble']['confidence_percent']:.2f}%")
    print(f"   Risk Level: {result['ensemble']['risk_level']}")
    
    print(f"\n📊 Individual Models:")
    print(f"   CatBoost:  {result['catboost']['probability']:.4f} ({result['catboost']['label']})")
    print(f"   LightGBM:  {result['lightgbm']['probability']:.4f} ({result['lightgbm']['label']})")
    
    # Example 2: Batch prediction
    print("\n" + "="*80)
    print("EXAMPLE 2: Batch Prediction")
    print("="*80)
    
    batch_requests = [
        {
            'attack_category': 'Network_Security',
            'attack_type': 'Network_Attack',
            'target_system': 'Embedded_System',
            'mitre_technique': 'Network_Intrusion',
            'packet_size': 0.2,
            'inter_arrival_time': 0.1,
            'packet_count_5s': 0.3,
            'mean_packet_size': 0.0,
            'spectral_entropy': 0.2,
            'frequency_band_energy': 0.1
        },
        {
            'attack_category': 'Access Control',
            'attack_type': 'IAM Misconfiguration',
            'target_system': 'Azure',
            'mitre_technique': 'Cloud_Access Control',
            'packet_size': 0.0,
            'inter_arrival_time': 0.0,
            'packet_count_5s': 0.0,
            'mean_packet_size': 0.0,
            'spectral_entropy': 0.0,
            'frequency_band_energy': 0.0
        }
    ]
    
    batch_results = detector.predict_batch(batch_requests)
    
    for i, result in enumerate(batch_results, 1):
        print(f"\nRequest {i}:")
        print(f"  Prediction: {result['ensemble']['label']}")
        print(f"  Confidence: {result['ensemble']['confidence_percent']:.2f}%")
        print(f"  Risk Level: {result['ensemble']['risk_level']}")
    
    # Example 3: Different thresholds
    print("\n" + "="*80)
    print("EXAMPLE 3: Custom Thresholds")
    print("="*80)
    
    thresholds = [0.3, 0.5, 0.7]
    
    for threshold in thresholds:
        result = detector.predict(sample_request, threshold=threshold)
        print(f"\nThreshold {threshold}: {result['ensemble']['label']} "
              f"(Confidence: {result['ensemble']['confidence_percent']:.2f}%)")
    
    print("\n" + "="*80)
    print("✓ Ensemble detector ready for deployment!")
    print("="*80 + "\n")
    
    # Show how to use in API
    print("="*80)
    print("FLASK/FASTAPI INTEGRATION EXAMPLE")
    print("="*80)
    print("""
# In your Flask/FastAPI application:

from ensemble_model import EnsembleWAFDetector

# Initialize once at startup
detector = EnsembleWAFDetector()

# In your API endpoint:
@app.post("/api/check-request")
def check_request(request_data: dict):
    # Extract features from request
    features = {
        'attack_category': request_data.get('category'),
        'attack_type': request_data.get('type'),
        'target_system': request_data.get('system'),
        'mitre_technique': request_data.get('technique'),
        'packet_size': request_data.get('packet_size'),
        'inter_arrival_time': request_data.get('inter_arrival_time'),
        'packet_count_5s': request_data.get('packet_count'),
        'mean_packet_size': request_data.get('mean_packet_size'),
        'spectral_entropy': request_data.get('entropy'),
        'frequency_band_energy': request_data.get('energy')
    }
    
    # Get prediction
    result = detector.predict(features)
    
    # Make decision
    if result['ensemble']['risk_level'] in ['CRITICAL', 'HIGH']:
        action = 'BLOCK'
    elif result['ensemble']['risk_level'] == 'MEDIUM':
        action = 'FLAG'
    else:
        action = 'ALLOW'
    
    return {
        'action': action,
        'prediction': result['ensemble']['label'],
        'confidence': result['ensemble']['confidence_percent'],
        'risk_level': result['ensemble']['risk_level']
    }
""")
    
    print("\n" + "="*80)
