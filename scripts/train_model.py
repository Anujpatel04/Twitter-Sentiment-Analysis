"""
Script to retrain the vectorizer and model together to ensure they match.
This will load the training data, train both together, and save them properly.
"""
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

print("Loading training data...")
# Load the training data
df = pd.read_csv('training.csv', encoding='latin_1', 
                 names=['target', 'ids', 'date', 'flag', 'user', 'text'])

print(f"Loaded {len(df)} samples")
print("Preprocessing data...")

# Convert target: 0 = negative, 4 = positive -> 0 = negative, 1 = positive
df['target'] = df['target'].replace(4, 1)

# Clean and prepare text data (convert to lowercase)
df['text'] = df['text'].astype(str).str.lower()

print("Creating vectorizer and transforming text...")
# Create and fit vectorizer on ALL data (to get full vocabulary)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['target']

print(f"Vectorizer created with {X.shape[1]} features")
print("Splitting data into train and test sets...")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Logistic Regression model...")
# Train the model with increased iterations for better convergence
model = LogisticRegression(max_iter=2000, random_state=42, solver='lbfgs')
model.fit(X_train, y_train)

print("Evaluating model...")
# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.4f}")

# Verify feature matching
print(f"\nVerification:")
print(f"Vectorizer features: {X.shape[1]}")
print(f"Model expects: {model.n_features_in_}")
print(f"Features match: {X.shape[1] == model.n_features_in_}")

# Save both files
print("\nSaving vectorizer and model...")
base_dir = os.path.dirname(os.path.abspath(__file__))
vectorizer_path = os.path.join(base_dir, 'vectorizer_correct.pkl')
model_path = os.path.join(base_dir, 'model_correct.joblib')

joblib.dump(vectorizer, vectorizer_path)
joblib.dump(model, model_path)

print(f"Vectorizer saved to: {vectorizer_path}")
print(f"Model saved to: {model_path}")
print("\nTraining complete! The vectorizer and model are now properly matched.")

