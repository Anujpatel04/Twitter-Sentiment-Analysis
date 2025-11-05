from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib
from joblib import load
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import os
import json

app = Flask(__name__)

# Load the properly matched vectorizer and model
base_dir = os.path.dirname(os.path.abspath(__file__))
model = load(os.path.join(base_dir, 'models', 'model_correct.joblib'))
vectorizer = joblib.load(os.path.join(base_dir, 'models', 'vectorizer_correct.pkl'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text_data = request.form.get('text')
    if not text_data or not text_data.strip():
        return render_template('result.html', 
                             sentiment="Error", 
                             confidence=0,
                             original_text="",
                             error="Please enter some text to analyze.")
    
    original_text = text_data.strip()
    # Preprocess: convert to lowercase (matching training data preprocessing)
    text_data_lower = [original_text.lower()]

    # Transform the input text data using the vectorizer
    text_data_transformed = vectorizer.transform(text_data_lower)

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

@app.route('/analytics')
def analytics():
    """Display analytics dashboard with model performance metrics"""
    # Load metrics if available
    metrics_path = os.path.join(base_dir, 'static', 'metrics.json')
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    else:
        # Default metrics if file doesn't exist
        metrics = {
            'accuracy': 79.99,
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'test_samples': 320000,
            'train_samples': 1280000
        }
    
    return render_template('analytics.html', metrics=metrics)

@app.route('/static/images/<path:filename>')
def serve_image(filename):
    """Serve generated images"""
    return send_from_directory(os.path.join(base_dir, 'static', 'images'), filename)


if __name__ == '__main__':
    # For local development
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    print("\n" + "="*50)
    print("Twitter Sentiment Analysis Server Starting...")
    print("="*50)
    print(f"Server running at: http://127.0.0.1:{port}/")
    print(f"Server running at: http://localhost:{port}/")
    print("Press CTRL+C to stop the server")
    print("="*50 + "\n")
    app.run(debug=debug, host='0.0.0.0', port=port)
