# 🐦 Twitter Sentiment Analysis - Capstone Project

A comprehensive machine learning project that analyzes sentiment from Twitter text data using Logistic Regression. This project includes a web application with a modern, professional interface for real-time sentiment analysis.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Screenshots](#screenshots)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

## 🎯 Overview

This project implements a sentiment analysis system that classifies Twitter text as either **Positive** or **Negative**. The model is trained on 1.6 million tweets using Logistic Regression with Count Vectorization, achieving **79.99% accuracy**.

### Key Highlights
- ✅ **79.99% Model Accuracy**
- ✅ **1.6M+ Training Samples**
- ✅ **684,358 Features**
- ✅ **Real-time Predictions**
- ✅ **Confidence Scores**
- ✅ **Professional Web Interface**
- ✅ **Model Performance Analytics**

## ✨ Features

### Web Application
- **Modern UI**: Clean, professional design with dark theme
- **Real-time Analysis**: Instant sentiment predictions
- **Confidence Scores**: Shows prediction confidence percentage
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Error Handling**: User-friendly error messages
- **Analytics Dashboard**: Visual representation of model performance

### Model Features
- **High Accuracy**: 79.99% accuracy on test set
- **Fast Inference**: Quick predictions with optimized preprocessing
- **Confidence Metrics**: Provides probability scores for each prediction
- **Robust Preprocessing**: Handles various text formats and edge cases

## 📁 Project Structure

```
TWITTER SENTIMENT ANALYSIS/
├── MLapp.py                  # Main Flask application
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
│
├── data/                     # Data directory
│   └── training.csv         # Training dataset (1.6M tweets)
│
├── models/                   # Trained models
│   ├── model_correct.joblib # Trained Logistic Regression model
│   └── vectorizer_correct.pkl # Fitted CountVectorizer
│
├── notebooks/                # Jupyter notebooks
│   └── exploring_twitter_sentiments_logisticreg.ipynb
│
├── scripts/                  # Utility scripts
│   ├── train_model.py       # Model training script
│   └── generate_metrics.py # Generate evaluation metrics
│
├── static/                   # Static files
│   ├── styles.css           # Main stylesheet
│   └── images/              # Generated graphs and images
│
└── templates/                # HTML templates
    ├── index.html           # Home page
    ├── result.html          # Results page
    └── analytics.html       # Analytics dashboard
```

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/twitter-sentiment-analysis.git
   cd twitter-sentiment-analysis
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**
   - The dataset should be placed in the `data/` directory
   - Ensure the file is named `training.csv`
   - Format: CSV with columns: target, ids, date, flag, user, text

5. **Train the model** (if not using pre-trained models)
   ```bash
   python scripts/train_model.py
   ```

## 💻 Usage

### Running the Application

1. **Start the Flask server**
   ```bash
   python MLapp.py
   ```

2. **Access the application**
   - Open your browser and navigate to: `http://localhost:5000`
   - Or: `http://127.0.0.1:5000`

3. **Using the Application**
   - Enter text in the text area
   - Click "Analyze Sentiment"
   - View the sentiment prediction (Positive/Negative) and confidence score

### Viewing Analytics

Access the analytics dashboard to see model performance metrics:
- Navigate to: `http://localhost:5000/analytics`

### API Endpoint (Future Enhancement)

```python
POST /api/predict
Content-Type: application/json

{
    "text": "I love this product!"
}

Response:
{
    "sentiment": "positive",
    "confidence": 92.5,
    "probabilities": {
        "negative": 7.5,
        "positive": 92.5
    }
}
```

## 📊 Model Performance

### Metrics
- **Accuracy**: 79.99%
- **Training Samples**: 1,600,000 tweets
- **Test Samples**: 320,000 tweets (20% split)
- **Features**: 684,358
- **Algorithm**: Logistic Regression
- **Solver**: LBFGS
- **Max Iterations**: 2000

### Evaluation Metrics
- **Precision**: Calculated on test set
- **Recall**: Calculated on test set
- **F1-Score**: Calculated on test set
- **Confusion Matrix**: Available in analytics dashboard

### View Detailed Metrics
Visit the `/analytics` route to see:
- Confusion Matrix
- Accuracy Trend
- Precision, Recall, F1-Score visualization
- ROC Curve (if available)
- Model performance comparison charts

## 🛠️ Technologies Used

### Backend
- **Python 3.9+**: Core programming language
- **Flask 3.0.0**: Web framework
- **scikit-learn 1.3.2**: Machine learning library
- **joblib 1.3.2**: Model serialization

### Frontend
- **HTML5**: Structure
- **CSS3**: Styling with modern design
- **JavaScript**: Interactive features
- **Inter Font**: Typography

### Machine Learning
- **Logistic Regression**: Classification algorithm
- **CountVectorizer**: Text feature extraction
- **Train-Test Split**: Data splitting (80-20)

### Data Processing
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations

### Analytics Dashboard
Visit `/analytics` to see interactive charts and metrics.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 👤 Author

**Anuj Patel**

- LinkedIn: https://www.linkedin.com/in/anujpatel04/
- Email: anuj.patel.29dec@gmail.com

## 🙏 Acknowledgments

- **Dataset**: Sentiment140 dataset from Kaggle
- **Scikit-learn**: For machine learning algorithms
- **Flask**: For web framework
- **Community**: For open-source libraries and resources

## 📞 Support

If you have any questions or encounter issues:
- Open an issue on GitHub
- Contact: anuj.patel.29dec@gmail.com

---

*This project demonstrates the application of machine learning in natural language processing for sentiment analysis.*

