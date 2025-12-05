"""
Example Integration - DDoS Detection in Web Application
This demonstrates how to integrate the DDoS detection model into your application
"""

import requests
import time
from datetime import datetime

class DDoSProtection:
    """
    DDoS Protection middleware for web applications
    """
    
    def __init__(self, api_url="http://localhost:5000"):
        self.api_url = api_url
        self.blocked_ips = set()
        self.alert_threshold = 0.8  # 80% confidence threshold
        
    def extract_features(self, request_data):
        """
        Extract network features from request
        In production, you would extract real network metrics
        """
        # Example feature extraction (customize based on your needs)
        features = {
            'flow_duration': request_data.get('duration', 0),
            'total_fwd_packets': request_data.get('packets', 0),
            'total_bwd_packets': request_data.get('response_packets', 0),
            'flow_bytes_per_sec': request_data.get('bytes_per_sec', 0),
            'flow_packets_per_sec': request_data.get('packets_per_sec', 0),
            'avg_packet_size': request_data.get('avg_packet_size', 0),
            # Add more features as needed
        }
        return features
    
    def check_request(self, request_data):
        """
        Check if request is a DDoS attack
        """
        # Extract features
        features = self.extract_features(request_data)
        
        # Check if IP is already blocked
        client_ip = request_data.get('ip', 'unknown')
        if client_ip in self.blocked_ips:
            return {
                'allowed': False,
                'reason': 'IP blocked due to previous DDoS detection',
                'ip': client_ip
            }
        
        try:
            # Call DDoS detection API
            response = requests.post(
                f"{self.api_url}/predict",
                json=features,
                timeout=1  # 1 second timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Check if DDoS attack detected
                if result['prediction'] == 'DDoS Attack':
                    confidence = result['confidence']
                    
                    if confidence >= self.alert_threshold:
                        # Block IP
                        self.blocked_ips.add(client_ip)
                        
                        # Log alert
                        self.log_alert(client_ip, confidence, features)
                        
                        return {
                            'allowed': False,
                            'reason': 'DDoS attack detected',
                            'confidence': confidence,
                            'ip': client_ip
                        }
                
                # Allow request
                return {
                    'allowed': True,
                    'confidence': result.get('normal_probability', 0),
                    'ip': client_ip
                }
            
            else:
                # API error - allow request but log warning
                print(f"Warning: DDoS API returned status {response.status_code}")
                return {'allowed': True, 'reason': 'API error'}
        
        except requests.exceptions.Timeout:
            # Timeout - allow request but log warning
            print("Warning: DDoS API timeout")
            return {'allowed': True, 'reason': 'API timeout'}
        
        except Exception as e:
            # Error - allow request but log error
            print(f"Error checking DDoS: {e}")
            return {'allowed': True, 'reason': 'Error'}
    
    def log_alert(self, ip, confidence, features):
        """
        Log DDoS alert
        """
        timestamp = datetime.now().isoformat()
        print(f"\n{'='*80}")
        print(f"🚨 DDoS ALERT - {timestamp}")
        print(f"{'='*80}")
        print(f"IP Address: {ip}")
        print(f"Confidence: {confidence*100:.2f}%")
        print(f"Features: {features}")
        print(f"{'='*80}\n")
        
        # In production, you would:
        # - Send email/SMS alert
        # - Log to security system
        # - Update firewall rules
        # - Notify security team
    
    def unblock_ip(self, ip):
        """
        Unblock an IP address
        """
        if ip in self.blocked_ips:
            self.blocked_ips.remove(ip)
            print(f"IP {ip} unblocked")
    
    def get_blocked_ips(self):
        """
        Get list of blocked IPs
        """
        return list(self.blocked_ips)


# Example 1: Flask Integration
def flask_example():
    """
    Example of integrating with Flask
    """
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    ddos_protection = DDoSProtection()
    
    @app.before_request
    def check_ddos():
        """
        Check every request for DDoS
        """
        request_data = {
            'ip': request.remote_addr,
            'duration': 100,  # Extract from request
            'packets': 50,    # Extract from request
            # Add more metrics
        }
        
        result = ddos_protection.check_request(request_data)
        
        if not result['allowed']:
            return jsonify({
                'error': 'Request blocked',
                'reason': result['reason']
            }), 403
    
    @app.route('/api/data')
    def get_data():
        return jsonify({'data': 'Your data here'})
    
    return app


# Example 2: Django Middleware
class DDoSMiddleware:
    """
    Django middleware for DDoS protection
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
        self.ddos_protection = DDoSProtection()
    
    def __call__(self, request):
        # Extract request data
        request_data = {
            'ip': self.get_client_ip(request),
            'duration': 100,  # Calculate from request
            'packets': 50,    # Extract from network layer
        }
        
        # Check for DDoS
        result = self.ddos_protection.check_request(request_data)
        
        if not result['allowed']:
            from django.http import HttpResponseForbidden
            return HttpResponseForbidden(f"Request blocked: {result['reason']}")
        
        # Continue processing
        response = self.get_response(request)
        return response
    
    def get_client_ip(self, request):
        """Get client IP from request"""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip


# Example 3: Standalone Usage
def standalone_example():
    """
    Example of standalone usage
    """
    print("DDoS Protection - Standalone Example")
    print("="*80)
    
    # Initialize protection
    protection = DDoSProtection()
    
    # Simulate incoming requests
    test_requests = [
        {
            'ip': '192.168.1.100',
            'duration': 1000,
            'packets': 10,
            'response_packets': 5,
            'bytes_per_sec': 5000,
            'packets_per_sec': 15,
            'avg_packet_size': 500
        },
        {
            'ip': '192.168.1.101',
            'duration': 50000,
            'packets': 10000,
            'response_packets': 100,
            'bytes_per_sec': 500000,
            'packets_per_sec': 1000,
            'avg_packet_size': 50
        }
    ]
    
    for i, request_data in enumerate(test_requests, 1):
        print(f"\nChecking Request #{i} from {request_data['ip']}...")
        result = protection.check_request(request_data)
        
        if result['allowed']:
            print(f"✅ Request ALLOWED")
        else:
            print(f"🚫 Request BLOCKED: {result['reason']}")
            print(f"   Confidence: {result.get('confidence', 0)*100:.2f}%")
    
    # Show blocked IPs
    print(f"\n\nBlocked IPs: {protection.get_blocked_ips()}")


# Example 4: Batch Processing
def batch_processing_example():
    """
    Example of batch processing multiple requests
    """
    print("DDoS Protection - Batch Processing Example")
    print("="*80)
    
    # Simulate batch of requests
    requests_batch = [
        {'ip': f'192.168.1.{i}', 'duration': 1000 * i, 'packets': 10 * i}
        for i in range(1, 11)
    ]
    
    # Prepare features for batch prediction
    features_batch = []
    for req in requests_batch:
        features = {
            'flow_duration': req['duration'],
            'total_fwd_packets': req['packets'],
            'total_bwd_packets': req['packets'] // 2,
            'flow_bytes_per_sec': req['duration'] * 5,
            'flow_packets_per_sec': req['packets'] / 10,
            'avg_packet_size': 500
        }
        features_batch.append(features)
    
    # Call batch prediction API
    try:
        response = requests.post(
            'http://localhost:5000/predict/batch',
            json={'samples': features_batch},
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\nProcessed {result['count']} requests")
            print(f"Total time: {result['total_time_ms']:.2f}ms")
            print(f"Average time per request: {result['avg_time_per_sample_ms']:.4f}ms")
            
            # Show results
            print("\nResults:")
            for r in result['results']:
                status = "🚫 BLOCK" if r['prediction'] == 'DDoS Attack' else "✅ ALLOW"
                print(f"  Request {r['sample_id']}: {status} (confidence: {r['confidence']*100:.1f}%)")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("="*80)
    print("DDoS Detection - Integration Examples")
    print("="*80)
    print("\nMake sure the API server is running: python api.py")
    print("\nSelect an example to run:")
    print("1. Standalone example")
    print("2. Batch processing example")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ")
    
    if choice == '1':
        standalone_example()
    elif choice == '2':
        batch_processing_example()
    else:
        print("Exiting...")
