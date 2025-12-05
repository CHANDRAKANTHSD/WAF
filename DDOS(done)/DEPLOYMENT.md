# Deployment Guide for Full-Stack Integration

## Overview

This guide explains how to integrate the trained DDoS detection model into your full-stack WAF application.

## Architecture Options

### Option 1: REST API (Recommended)

Create a Flask/FastAPI service that serves predictions.

#### Flask Example

```python
from flask import Flask, request, jsonify
from inference import DDoSInference
import pandas as pd

app = Flask(__name__)

# Initialize model once at startup
detector = DDoSInference(
    model_path='LightGBM_ddos_model.joblib',
    scaler_path='scaler.joblib',
    features_path='selected_features.pkl'
)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Make prediction
        result = detector.predict_single(df)
        
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'ddos_probability': result['ddos_probability'],
            'inference_time_ms': result['inference_time_ms']
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

#### FastAPI Example

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from inference import DDoSInference
import pandas as pd

app = FastAPI()

# Initialize model
detector = DDoSInference(
    model_path='LightGBM_ddos_model.joblib',
    scaler_path='scaler.joblib',
    features_path='selected_features.pkl'
)

class NetworkTraffic(BaseModel):
    features: dict

@app.post("/predict")
async def predict(traffic: NetworkTraffic):
    try:
        df = pd.DataFrame([traffic.features])
        result = detector.predict_single(df)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

### Option 2: Direct Integration

Import the inference module directly in your Python application.

```python
from inference import DDoSInference
import pandas as pd

# Initialize once
detector = DDoSInference(
    model_path='LightGBM_ddos_model.joblib',
    scaler_path='scaler.joblib',
    features_path='selected_features.pkl'
)

# Use in your application
def check_traffic(network_features):
    result = detector.predict_single(network_features)
    
    if result['prediction'] == 'DDoS Attack':
        # Trigger alert
        alert_security_team(result)
        # Block traffic
        block_ip(network_features['source_ip'])
    
    return result
```

### Option 3: Microservice with Docker

Create a containerized service for easy deployment.

#### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inference.py .
COPY config.py .
COPY *.joblib .
COPY *.pkl .
COPY api.py .

EXPOSE 5000

CMD ["python", "api.py"]
```

#### docker-compose.yml

```yaml
version: '3.8'

services:
  ddos-detector:
    build: .
    ports:
      - "5000:5000"
    environment:
      - MODEL_PATH=LightGBM_ddos_model.joblib
      - SCALER_PATH=scaler.joblib
      - FEATURES_PATH=selected_features.pkl
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Performance Optimization

### 1. Model Loading

Load the model once at application startup, not per request:

```python
# Good - Load once
detector = DDoSInference(...)

# Bad - Don't do this
def predict(data):
    detector = DDoSInference(...)  # Slow!
    return detector.predict(data)
```

### 2. Batch Predictions

Process multiple samples together for better throughput:

```python
# Process batch of 100 samples
predictions, probs, time = detector.predict(df_batch)
```

### 3. Caching

Cache predictions for identical feature sets:

```python
from functools import lru_cache
import hashlib
import json

@lru_cache(maxsize=1000)
def cached_predict(features_hash):
    features = json.loads(features_hash)
    return detector.predict_single(features)

def predict_with_cache(features):
    features_hash = hashlib.md5(
        json.dumps(features, sort_keys=True).encode()
    ).hexdigest()
    return cached_predict(features_hash)
```

## Monitoring

### Key Metrics to Track

1. **Prediction Latency**: Time to make predictions
2. **Throughput**: Predictions per second
3. **Accuracy**: Compare predictions with actual attacks
4. **False Positive Rate**: Normal traffic flagged as DDoS
5. **False Negative Rate**: DDoS attacks missed

### Example Monitoring Code

```python
import time
from collections import deque
import statistics

class ModelMonitor:
    def __init__(self, window_size=1000):
        self.predictions = deque(maxlen=window_size)
        self.latencies = deque(maxlen=window_size)
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.true_negatives = 0
    
    def log_prediction(self, prediction, actual, latency):
        self.predictions.append(prediction)
        self.latencies.append(latency)
        
        if actual is not None:
            if prediction == 1 and actual == 1:
                self.true_positives += 1
            elif prediction == 1 and actual == 0:
                self.false_positives += 1
            elif prediction == 0 and actual == 1:
                self.false_negatives += 1
            else:
                self.true_negatives += 1
    
    def get_metrics(self):
        return {
            'avg_latency_ms': statistics.mean(self.latencies),
            'p95_latency_ms': statistics.quantiles(self.latencies, n=20)[18],
            'predictions_count': len(self.predictions),
            'ddos_rate': sum(self.predictions) / len(self.predictions),
            'accuracy': (self.true_positives + self.true_negatives) / 
                       (self.true_positives + self.false_positives + 
                        self.false_negatives + self.true_negatives)
        }
```

## Security Considerations

1. **Input Validation**: Validate all input features
2. **Rate Limiting**: Prevent API abuse
3. **Authentication**: Secure API endpoints
4. **Logging**: Log all predictions for audit
5. **Model Updates**: Plan for periodic retraining

## Scaling

### Horizontal Scaling

Deploy multiple instances behind a load balancer:

```yaml
# Kubernetes deployment example
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ddos-detector
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ddos-detector
  template:
    metadata:
      labels:
        app: ddos-detector
    spec:
      containers:
      - name: detector
        image: ddos-detector:latest
        ports:
        - containerPort: 5000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

### Vertical Scaling

Optimize for single-instance performance:
- Use faster CPU
- Increase memory for larger batches
- Use GPU for XGBoost (if applicable)

## Troubleshooting

### High Latency
- Check batch size
- Profile code for bottlenecks
- Consider model quantization

### High Memory Usage
- Reduce batch size
- Use model compression
- Monitor for memory leaks

### Accuracy Degradation
- Retrain with recent data
- Check for data drift
- Validate feature distributions

## Next Steps

1. Set up monitoring and alerting
2. Implement A/B testing for model updates
3. Create automated retraining pipeline
4. Document API endpoints
5. Set up CI/CD for model deployment
