"""
Flask API for DDoS Detection Model
Simple REST API for serving predictions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from inference import DDoSInference
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Initialize model (load once at startup)
logger.info("Loading DDoS detection model...")
try:
    detector = DDoSInference(
        model_path='XGBoost_ddos_model.joblib',  # XGBoost was selected as best
        scaler_path='scaler.joblib',
        features_path='selected_features.pkl'
    )
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    detector = None

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'service': 'DDoS Detection API',
        'version': '1.0',
        'status': 'running',
        'endpoints': {
            '/health': 'Health check',
            '/predict': 'Single prediction (POST)',
            '/predict/batch': 'Batch predictions (POST)',
            '/model/info': 'Model information'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    if detector is None:
        return jsonify({
            'status': 'unhealthy',
            'message': 'Model not loaded'
        }), 503
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'timestamp': time.time()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Single prediction endpoint"""
    if detector is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 503
    
    try:
        # Get data from request
        data = request.json
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Make prediction
        start_time = time.time()
        result = detector.predict_single(data)
        total_time = (time.time() - start_time) * 1000
        
        # Add request processing time
        result['total_time_ms'] = total_time
        result['success'] = True
        
        logger.info(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.2%})")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint"""
    if detector is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 503
    
    try:
        # Get data from request
        data = request.json
        
        if not data or 'samples' not in data:
            return jsonify({
                'success': False,
                'error': 'No samples provided. Expected format: {"samples": [...]}'
            }), 400
        
        samples = data['samples']
        
        if not isinstance(samples, list):
            return jsonify({
                'success': False,
                'error': 'Samples must be a list'
            }), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(samples)
        
        # Make predictions
        start_time = time.time()
        predictions, probabilities, inference_time = detector.predict(df)
        total_time = (time.time() - start_time) * 1000
        
        # Format results
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                'sample_id': i,
                'prediction': 'DDoS Attack' if pred == 1 else 'Normal',
                'confidence': float(prob[pred]),
                'ddos_probability': float(prob[1]),
                'normal_probability': float(prob[0])
            })
        
        logger.info(f"Batch prediction: {len(samples)} samples in {total_time:.2f}ms")
        
        return jsonify({
            'success': True,
            'count': len(samples),
            'results': results,
            'inference_time_ms': inference_time,
            'total_time_ms': total_time,
            'avg_time_per_sample_ms': total_time / len(samples)
        })
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    if detector is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 503
    
    try:
        return jsonify({
            'success': True,
            'model_type': type(detector.model).__name__,
            'n_features': len(detector.selected_features),
            'features': detector.selected_features[:10],  # First 10 features
            'scaler_type': type(detector.scaler).__name__
        })
    
    except Exception as e:
        logger.error(f"Model info error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal error: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    # Run the app
    logger.info("Starting DDoS Detection API server...")
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False  # Set to True for development
    )
