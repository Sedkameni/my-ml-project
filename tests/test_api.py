import pytest
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from appproj_fpa import app, load_models

@pytest.fixture
def client():
    """Create a test client"""
    app.config['TESTING'] = True
    load_models()
    with app.test_client() as client:
        yield client

def test_home_endpoint(client):
    """Test the home endpoint"""
    response = client.get('/')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'running'
    assert 'model_version' in data

def test_health_endpoint(client):
    """Test the health check endpoint"""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'
    assert data['model_loaded'] == True
    assert data['vectorizer_loaded'] == True

def test_predict_positive_sentiment(client):
    """Test prediction with positive tweet"""
    response = client.post('/predict',
                          data=json.dumps({'tweet': 'I love this amazing product! Best day ever!'}),
                          content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'sentiment' in data
    assert data['sentiment'] in ['positive', 'negative']
    assert 'confidence' in data
    assert 0 <= data['confidence'] <= 1

def test_predict_negative_sentiment(client):
    """Test prediction with negative tweet"""
    response = client.post('/predict',
                          data=json.dumps({'tweet': 'I hate this terrible product. Worst experience ever.'}),
                          content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'sentiment' in data
    assert data['sentiment'] in ['positive', 'negative']
    assert 'confidence' in data

def test_predict_no_tweet(client):
    """Test prediction with no tweet provided"""
    response = client.post('/predict',
                          data=json.dumps({}),
                          content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_predict_empty_tweet(client):
    """Test prediction with empty tweet"""
    response = client.post('/predict',
                          data=json.dumps({'tweet': ''}),
                          content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_predict_special_characters(client):
    """Test prediction with tweet containing special characters"""
    response = client.post('/predict',
                          data=json.dumps({'tweet': '@user Check out this link: https://example.com #awesome'}),
                          content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'cleaned_tweet' in data
    assert 'sentiment' in data

def test_batch_predict(client):
    """Test batch prediction"""
    tweets = [
        'I love this product!',
        'This is terrible.',
        'Amazing experience today!'
    ]
    response = client.post('/batch_predict',
                          data=json.dumps({'tweets': tweets}),
                          content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'results' in data
    assert len(data['results']) == 3
    assert data['count'] == 3

def test_batch_predict_empty_array(client):
    """Test batch prediction with empty array"""
    response = client.post('/batch_predict',
                          data=json.dumps({'tweets': []}),
                          content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_batch_predict_no_tweets(client):
    """Test batch prediction with no tweets field"""
    response = client.post('/batch_predict',
                          data=json.dumps({}),
                          content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_response_structure(client):
    """Test that response has correct structure"""
    response = client.post('/predict',
                          data=json.dumps({'tweet': 'This is a test tweet'}),
                          content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    
    # Check all required fields
    required_fields = ['tweet', 'cleaned_tweet', 'sentiment', 'confidence', 'probabilities', 'model_version']
    for field in required_fields:
        assert field in data
    
    # Check probabilities structure
    assert 'negative' in data['probabilities']
    assert 'positive' in data['probabilities']
    
    # Check probability values sum to approximately 1
    prob_sum = data['probabilities']['negative'] + data['probabilities']['positive']
    assert abs(prob_sum - 1.0) < 0.01

if __name__ == '__main__':
    pytest.main([__file__, '-v'])