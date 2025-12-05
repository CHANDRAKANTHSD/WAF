"""
Ensemble WAF REST API
Flask API for deploying the ensemble model in production
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime
from ensemble_waf import EnsembleWAF
import json

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Ensemble WAF
waf = None

def init_waf(strategy='cascading'):
    """Initialize the WAF models"""
    global waf
    try:
        waf = EnsembleWAF(strategy=strategy)
        waf.load_models()
        logger.info(f"✅ Ensemble WAF initialized with strategy: {strategy}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize WAF: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if waf is None:
        return jsonify({
            'status': 'unhealthy',
            'message': 'WAF not initialized'
        }), 503
    
    return jsonify({
        'status': 'healthy',
        'strategy': waf.strategy,
        'models_loaded': {
            'xgboost': waf.xgboost_model is not None,
            'cnn_bilstm': waf.cnn_bilstm_model is not None
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/check', methods=['POST'])
def check_request():
    """
    Check if a request is malicious
    
    Request body:
    {
        "url": "/path/to/resource",
        "method": "GET",
        "headers": {...},
        "body": "...",
        "type": "http|network|packet"
    }
    
    Response:
    {
        "is_attack": true/false,
        "confidence": 0.95,
        "action": "block|allow",
        "details": {...}
    }
    """
    if waf is None:
        return jsonify({
            'error': 'WAF not initialized'
        }), 503
    
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided'
            }), 400
        
        # Predict
        result = waf.predict(data)
        
        # Log the decision
        logger.info(f"Request checked: {data.get('url', 'N/A')} - "
                   f"{'BLOCKED' if result['is_attack'] else 'ALLOWED'} "
                   f"(confidence: {result['confidence']:.2%})")
        
        # Return response
        return jsonify({
            'is_attack': result['is_attack'],
            'confidence': result['confidence'],
            'action': 'block' if result['is_attack'] else 'allow',
            'details': {
                'strategy': result['strategy'],
                'decision_maker': result['decision_maker'],
                'inference_time_ms': result['inference_time_ms'],
                'xgboost': {
                    'prediction': result.get('xgboost_prediction'),
                    'confidence': result.get('xgboost_confidence')
                } if result.get('xgboost_confidence') is not None else None,
                'cnn_bilstm': {
                    'prediction': result.get('cnn_prediction'),
                    'confidence': result.get('cnn_confidence')
                } if result.get('cnn_confidence') is not None else None
            },
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/batch', methods=['POST'])
def batch_check():
    """
    Check multiple requests at once
    
    Request body:
    {
        "requests": [
            {"url": "/path1", "method": "GET"},
            {"url": "/path2", "method": "POST"}
        ]
    }
    """
    if waf is None:
        return jsonify({
            'error': 'WAF not initialized'
        }), 503
    
    try:
        data = request.get_json()
        requests_to_check = data.get('requests', [])
        
        if not requests_to_check:
            return jsonify({
                'error': 'No requests provided'
            }), 400
        
        results = []
        for req_data in requests_to_check:
            result = waf.predict(req_data)
            results.append({
                'url': req_data.get('url', 'N/A'),
                'is_attack': result['is_attack'],
                'confidence': result['confidence'],
                'action': 'block' if result['is_attack'] else 'allow'
            })
        
        return jsonify({
            'results': results,
            'total': len(results),
            'attacks_detected': sum(1 for r in results if r['is_attack']),
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get WAF statistics"""
    if waf is None:
        return jsonify({
            'error': 'WAF not initialized'
        }), 503
    
    stats = waf.get_stats()
    return jsonify({
        'statistics': stats,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/stats/reset', methods=['POST'])
def reset_stats():
    """Reset WAF statistics"""
    if waf is None:
        return jsonify({
            'error': 'WAF not initialized'
        }), 503
    
    waf.reset_stats()
    return jsonify({
        'message': 'Statistics reset successfully',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/config', methods=['GET'])
def get_config():
    """Get current WAF configuration"""
    if waf is None:
        return jsonify({
            'error': 'WAF not initialized'
        }), 503
    
    return jsonify({
        'strategy': waf.strategy,
        'models': {
            'xgboost': {
                'loaded': waf.xgboost_model is not None,
                'features': len(waf.xgboost_features) if waf.xgboost_features else 0
            },
            'cnn_bilstm': {
                'loaded': waf.cnn_bilstm_model is not None,
                'max_length': waf.cnn_max_length if waf.cnn_max_length else 0
            }
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/config/strategy', methods=['POST'])
def update_strategy():
    """
    Update ensemble strategy
    
    Request body:
    {
        "strategy": "parallel|weighted|cascading"
    }
    """
    if waf is None:
        return jsonify({
            'error': 'WAF not initialized'
        }), 503
    
    try:
        data = request.get_json()
        new_strategy = data.get('strategy')
        
        if new_strategy not in ['parallel', 'weighted', 'cascading']:
            return jsonify({
                'error': 'Invalid strategy. Must be: parallel, weighted, or cascading'
            }), 400
        
        waf.strategy = new_strategy
        logger.info(f"Strategy updated to: {new_strategy}")
        
        return jsonify({
            'message': f'Strategy updated to {new_strategy}',
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error updating strategy: {e}")
        return jsonify({
            'error': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Ensemble WAF API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--strategy', default='cascading', 
                       choices=['parallel', 'weighted', 'cascading'],
                       help='Ensemble strategy')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Initialize WAF
    print("="*60)
    print("Ensemble WAF API Server")
    print("="*60)
    
    if init_waf(strategy=args.strategy):
        print(f"\n✅ WAF initialized successfully")
        print(f"   Strategy: {args.strategy}")
        print(f"   Host: {args.host}")
        print(f"   Port: {args.port}")
        print(f"\n🚀 Starting server...")
        print("="*60)
        
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug
        )
    else:
        print("\n❌ Failed to initialize WAF")
        print("   Make sure models are trained and saved!")
        exit(1)
