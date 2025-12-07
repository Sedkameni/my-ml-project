# app.py
from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

# Initialize Flask
app = Flask(__name__)

# Load or train model
if not os.path.exists("logistic_model.pkl"):
    print(" Model file not found. Training new model...")
    import subprocess

    result = subprocess.run(["python", "logistic_model.py"])
    if result.returncode != 0:
        print("✗ Error: Failed to train model")
        raise RuntimeError("Model training failed")

try:
    with open("logistic_model.pkl", "rb") as f:
        loaded_data = pickle.load(f)

        # Handle both simple model and model with metadata
        if isinstance(loaded_data, dict):
            # Enhanced format with metadata
            model = loaded_data['model']
            feature_names = loaded_data.get('feature_names',
                                            ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
            species = loaded_data.get('target_names',
                                      ['setosa', 'versicolor', 'virginica'])
            accuracy = loaded_data.get('accuracy', None)
            print("✓ Model loaded successfully (with metadata)")
            if accuracy:
                print(f"  Model accuracy: {accuracy:.4f}")
        else:
            # Simple format - just the model
            model = loaded_data
            feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            species = ['setosa', 'versicolor', 'virginica']
            print("✓ Model loaded successfully")

except Exception as e:
    print(f" Error loading model: {e}")
    raise


@app.route('/')
def home():
    """Home endpoint with API documentation"""
    return jsonify({
        'message': 'Iris Logistic Regression API',
        'version': '1.0',
        'endpoints': {
            '/': 'GET - API documentation (this page)',
            '/predict': 'POST - Make predictions',
            '/health': 'GET - Health check'
        },
        'usage': 'Send POST request to /predict with iris features',
        'example_request': {
            'sepal_length': 5.1,
            'sepal_width': 3.5,
            'petal_length': 1.4,
            'petal_width': 0.2
        },
        'example_response': {
            'prediction': 0,
            'species': 'setosa',
            'confidence': 'high'
        }
    })


@app.route('/health')
def health():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'species_classes': species
    })


@app.route('/predict', methods=['POST'])
def predict_iris():
    """
    Predict iris species from flower measurements

    Expected JSON format:
    {
        "sepal_length": float,
        "sepal_width": float,
        "petal_length": float,
        "petal_width": float
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()

        if not data:
            return jsonify({
                'error': 'No data provided',
                'hint': 'Send JSON with sepal_length, sepal_width, petal_length, petal_width'
            }), 400

        # Validate input fields
        required_fields = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}',
                'required_fields': required_fields,
                'received_fields': list(data.keys())
            }), 400

        # Validate data types
        try:
            features = np.array([[
                float(data['sepal_length']),
                float(data['sepal_width']),
                float(data['petal_length']),
                float(data['petal_width'])
            ]])
        except (ValueError, TypeError) as e:
            return jsonify({
                'error': 'Invalid data type',
                'hint': 'All values must be numeric',
                'details': str(e)
            }), 400

        # Validate feature ranges (basic sanity check)
        if np.any(features < 0) or np.any(features > 20):
            return jsonify({
                'warning': 'Feature values outside typical range (0-20)',
                'note': 'Prediction may be unreliable'
            })

        # Make prediction
        prediction = model.predict(features)
        prediction_proba = model.predict_proba(features)

        predicted_class = int(prediction[0])
        predicted_species = species[predicted_class]
        confidence = float(np.max(prediction_proba))

        # Prepare response
        response = {
            'prediction': predicted_class,
            'species': predicted_species,
            'confidence': round(confidence, 4),
            'probabilities': {
                species[i]: round(float(prediction_proba[0][i]), 4)
                for i in range(len(species))
            },
            'input': {
                'sepal_length': float(data['sepal_length']),
                'sepal_width': float(data['sepal_width']),
                'petal_length': float(data['petal_length']),
                'petal_width': float(data['petal_width'])
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e)
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)