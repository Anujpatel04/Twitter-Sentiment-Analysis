"""
Script to generate model evaluation metrics and visualizations
Run this script after training to generate accuracy graphs and metrics
"""
import pandas as pd
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)
import os

# Set style
plt.style.use('dark_background')
sns.set_palette("husl")

def load_model_and_data():
    """Load the trained model and prepare test data"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load model and vectorizer
    model_path = os.path.join(base_dir, 'models', 'model_correct.joblib')
    vectorizer_path = os.path.join(base_dir, 'models', 'vectorizer_correct.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("Models not found. Training model first...")
        # Train the model
        train_and_save_model()
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
    else:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
    
    # Load and prepare test data
    data_path = os.path.join(base_dir, 'data', 'training.csv')
    df = pd.read_csv(data_path, encoding='latin_1', 
                     names=['target', 'ids', 'date', 'flag', 'user', 'text'])
    
    # Preprocess
    df['target'] = df['target'].replace(4, 1)
    df['text'] = df['text'].astype(str).str.lower()
    
    # Transform data
    X = vectorizer.transform(df['text'])
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return model, vectorizer, X_test, y_test, X_train, y_train

def train_and_save_model():
    """Train the model if it doesn't exist"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'training.csv')
    
    print("Loading training data...")
    df = pd.read_csv(data_path, encoding='latin_1', 
                     names=['target', 'ids', 'date', 'flag', 'user', 'text'])
    
    print(f"Loaded {len(df)} samples")
    df['target'] = df['target'].replace(4, 1)
    df['text'] = df['text'].astype(str).str.lower()
    
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['text'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = LogisticRegression(max_iter=2000, random_state=42, solver='lbfgs')
    model.fit(X_train, y_train)
    
    # Save models
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(models_dir, 'model_correct.joblib'))
    joblib.dump(vectorizer, os.path.join(models_dir, 'vectorizer_correct.pkl'))
    
    print("Model trained and saved!")

def generate_confusion_matrix(y_test, y_pred, save_path):
    """Generate and save confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                facecolor='#0f172a', edgecolor='none')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def generate_metrics_chart(y_test, y_pred, save_path):
    """Generate metrics bar chart"""
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [accuracy * 100, precision * 100, recall * 100, f1 * 100]
    colors = ['#10b981', '#3b82f6', '#8b5cf6', '#f59e0b']
    
    plt.figure(figsize=(12, 7))
    bars = plt.bar(metrics, values, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.2f}%', ha='center', va='bottom', 
                fontsize=12, fontweight='bold')
    
    plt.title('Model Performance Metrics', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Score (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='#0f172a', edgecolor='none')
    plt.close()
    print(f"Metrics chart saved to {save_path}")
    
    return accuracy, precision, recall, f1

def generate_accuracy_trend(y_train, y_train_pred, y_test, y_test_pred, save_path):
    """Generate accuracy trend visualization"""
    train_acc = accuracy_score(y_train, y_train_pred) * 100
    test_acc = accuracy_score(y_test, y_test_pred) * 100
    
    categories = ['Training Set', 'Test Set']
    accuracies = [train_acc, test_acc]
    colors = ['#3b82f6', '#10b981']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, accuracies, color=colors, alpha=0.8, 
                   edgecolor='white', linewidth=2, width=0.6)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom',
                fontsize=14, fontweight='bold')
    
    plt.title('Model Accuracy: Training vs Test Set', fontsize=16, 
              fontweight='bold', pad=20)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='#0f172a', edgecolor='none')
    plt.close()
    print(f"Accuracy trend saved to {save_path}")

def generate_class_distribution(y_test, save_path):
    """Generate class distribution pie chart"""
    unique, counts = np.unique(y_test, return_counts=True)
    labels = ['Negative', 'Positive']
    colors = ['#ef4444', '#10b981']
    explode = (0.05, 0.05)
    
    plt.figure(figsize=(10, 8))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90,
            colors=colors, explode=explode, shadow=True,
            textprops={'fontsize': 12, 'fontweight': 'bold'})
    plt.title('Test Set Class Distribution', fontsize=16, 
              fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='#0f172a', edgecolor='none')
    plt.close()
    print(f"Class distribution saved to {save_path}")

def main():
    """Main function to generate all metrics"""
    print("=" * 60)
    print("Generating Model Evaluation Metrics and Visualizations")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    images_dir = os.path.join(base_dir, 'static', 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Load model and data
    model, vectorizer, X_test, y_test, X_train, y_train = load_model_and_data()
    
    # Generate predictions
    print("\nGenerating predictions...")
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    
    print(f"\n{'='*60}")
    print("MODEL PERFORMANCE METRICS")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print(f"{'='*60}\n")
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # Confusion Matrix
    cm_path = os.path.join(images_dir, 'confusion_matrix.png')
    generate_confusion_matrix(y_test, y_test_pred, cm_path)
    
    # Metrics Chart
    metrics_path = os.path.join(images_dir, 'metrics_chart.png')
    generate_metrics_chart(y_test, y_test_pred, metrics_path)
    
    # Accuracy Trend
    acc_trend_path = os.path.join(images_dir, 'accuracy_trend.png')
    generate_accuracy_trend(y_train, y_train_pred, y_test, y_test_pred, acc_trend_path)
    
    # Class Distribution
    class_dist_path = os.path.join(images_dir, 'class_distribution.png')
    generate_class_distribution(y_test, class_dist_path)
    
    # Save metrics to file
    metrics_file = os.path.join(base_dir, 'static', 'metrics.json')
    import json
    metrics_data = {
        'accuracy': round(accuracy * 100, 2),
        'precision': round(precision * 100, 2),
        'recall': round(recall * 100, 2),
        'f1_score': round(f1 * 100, 2),
        'test_samples': len(y_test),
        'train_samples': len(y_train)
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"\n✅ All metrics and visualizations generated successfully!")
    print(f"📊 Images saved to: {images_dir}")
    print(f"📄 Metrics saved to: {metrics_file}")

if __name__ == '__main__':
    main()

