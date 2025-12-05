"""
Test client for DDoS Detection API
Demonstrates how to use the API endpoints
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("Testing Health Endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200

def test_single_prediction():
    """Test single prediction endpoint"""
    print("\n" + "="*60)
    print("Testing Single Prediction")
    print("="*60)
    
    # Example network traffic features (dummy data)
    sample = {
        'flow_duration': 1000,
        'total_fwd_packets': 10,
        'total_bwd_packets': 5,
        'flow_bytes_per_sec': 5000,
        'flow_packets_per_sec': 15,
        'avg_packet_size': 500
    }
    
    print(f"Sending sample: {json.dumps(sample, indent=2)}")
    
    start_time = time.time()
    response = requests.post(
        f"{BASE_URL}/predict",
        json=sample,
        headers={'Content-Type': 'application/json'}
    )
    request_time = (time.time() - start_time) * 1000
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"Request Time: {request_time:.2f}ms")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n" + "="*60)
    print("Testing Batch Prediction")
    print("="*60)
    
    # Example batch of network traffic (dummy data)
    samples = [
        {
            'flow_duration': 1000,
            'total_fwd_packets': 10,
            'total_bwd_packets': 5
        },
        {
            'flow_duration': 5000,
            'total_fwd_packets': 100,
            'total_bwd_packets': 50
        },
        {
            'flow_duration': 500,
            'total_fwd_packets': 5,
            'total_bwd_packets': 2
        }
    ]
    
    print(f"Sending {len(samples)} samples")
    
    start_time = time.time()
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        json={'samples': samples},
        headers={'Content-Type': 'application/json'}
    )
    request_time = (time.time() - start_time) * 1000
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"Request Time: {request_time:.2f}ms")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200

def test_model_info():
    """Test model info endpoint"""
    print("\n" + "="*60)
    print("Testing Model Info")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200

def main():
    """Run all tests"""
    print("="*60)
    print("DDoS Detection API - Test Client")
    print("="*60)
    print("\nMake sure the API server is running:")
    print("  python api.py")
    print("\nPress Enter to continue...")
    input()
    
    results = []
    
    try:
        # Test health
        results.append(("Health Check", test_health()))
        
        # Test model info
        results.append(("Model Info", test_model_info()))
        
        # Test single prediction
        results.append(("Single Prediction", test_single_prediction()))
        
        # Test batch prediction
        results.append(("Batch Prediction", test_batch_prediction()))
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API server")
        print("Make sure the server is running: python api.py")
        return
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:25s} {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed")

if __name__ == "__main__":
    main()
