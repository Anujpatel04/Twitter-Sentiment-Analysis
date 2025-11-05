from flask import Flask, request, jsonify, render_template
import joblib
from joblib import load
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

app = Flask(__name__)

# Load the properly matched vectorizer and model
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
model = load(os.path.join(base_dir, 'models', 'model_correct.joblib'))
vectorizer = joblib.load(os.path.join(base_dir, 'models', 'vectorizer_correct.pkl'))

# Model metadata
MODEL_ACCURACY = 79.99
TRAINING_SAMPLES = 1280000
FEATURE_COUNT = 684358


@app.route('/')
def index():
    return render_template('index.html', 
                         accuracy=MODEL_ACCURACY,
                         training_samples=TRAINING_SAMPLES,
                         feature_count=FEATURE_COUNT)

@app.route('/predict', methods=['POST'])
def predict():
    text_data = request.form.get('text')
    if not text_data:
        return render_template('result.html', 
                             sentiment="Error", 
                             confidence=0,
                             original_text="No text provided",
                             error="Please enter some text to analyze.")
    
    original_text = text_data
    # Preprocess: convert to lowercase (matching training data preprocessing)
    text_data = [text_data.lower()]

    # Transform the input text data using the vectorizer
    text_data_transformed = vectorizer.transform(text_data)

    # Predict the sentiment using the pre-trained model
    prediction = model.predict(text_data_transformed)
    
    # Get prediction probabilities for confidence score
    probabilities = model.predict_proba(text_data_transformed)[0]
    confidence = max(probabilities) * 100

    # Convert the prediction to human-readable sentiment
    # Model outputs: 0 = negative, 1 = positive
    sentiment = "Positive" if prediction[0] == 1 else "Negative"

    return render_template('result.html', 
                         sentiment=sentiment,
                         confidence=round(confidence, 2),
                         original_text=original_text)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint for JSON responses"""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Please provide text in JSON format'}), 400
    
    text_data = data['text']
    text_data_lower = [text_data.lower()]
    
    text_transformed = vectorizer.transform(text_data_lower)
    prediction = model.predict(text_transformed)[0]
    probabilities = model.predict_proba(text_transformed)[0]
    confidence = max(probabilities) * 100
    
    result = {
        'text': text_data,
        'sentiment': 'positive' if prediction == 1 else 'negative',
        'confidence': round(confidence, 2),
        'probabilities': {
            'negative': round(probabilities[0] * 100, 2),
            'positive': round(probabilities[1] * 100, 2)
        }
    }
    
    return jsonify(result)


if __name__ == '__main__':
    print("\n" + "="*50)
    print("Twitter Sentiment Analysis Server Starting...")
    print("="*50)
    print("Server running at: http://127.0.0.1:5000/")
    print("Server running at: http://localhost:5000/")
    print("Press CTRL+C to stop the server")
    print("="*50 + "\n")
    app.run(debug=True, host='127.0.0.1', port=5000)
