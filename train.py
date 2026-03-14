"""
train.py - Sentiment Analysis Model Training Script
Based directly on Model_building.ipynb

This script implements a complete machine learning pipeline for sentiment analysis:
1. Data loading and preprocessing
2. Feature engineering with TF-IDF
3. Training multiple models for comparison
4. Selecting best model (Logistic Regression)
5. Hyperparameter tuning with GridSearchCV
6. Saving the final tuned model
"""

# ==================== IMPORT LIBRARIES ====================
import pandas as pd
import numpy as np
import re
import string
import pickle
from collections import Counter
from pathlib import Path

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import nltk
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
print("="*80)
print("SENTIMENT ANALYSIS MODEL TRAINING - COMPLETE PIPELINE")
print("="*80)

# Using relative paths for portability
# The script can run from any location as long as files are in correct relative paths
DATA_PATH = "IMDB Dataset.csv"  # Dataset in root directory
MODEL_DIR = Path("../static/model")  # Save models in static folder (matches notebook)

# Create directory if it doesn't exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
print(f"\nModel files will be saved to: {MODEL_DIR.absolute()}")

# ==================== DOWNLOAD NLTK DATA ====================
print("\n" + "="*80)
print("STEP 1: DOWNLOADING NLTK DATA")
print("="*80)

# Download NLTK data if not already present
# This ensures stopwords are available for preprocessing
nltk.download('stopwords', download_dir='../static/model', quiet=True)
nltk.download('punkt', quiet=True)
print("NLTK data downloaded")

# ==================== LOAD DATA ====================
print("\n" + "="*80)
print("STEP 2: LOADING DATASET")
print("="*80)

# Using pandas for data manipulation
# IMDb dataset is balanced (50k reviews, 25k positive, 25k negative)
try:
    data = pd.read_csv(DATA_PATH)
    print(f"Original shape: {data.shape}")
except FileNotFoundError:
    print(f"ERROR: Could not find {DATA_PATH}")
    print("Please ensure the IMDb dataset is in the current directory.")
    exit(1)

# ==================== REMOVE DUPLICATES ====================
print("\n" + "="*80)
print("STEP 3: REMOVING DUPLICATES")
print("="*80)

# Remove duplicate reviews to prevent data leakage
# Duplicate reviews could cause model to overfit on repeated content
duplicates = data.duplicated().sum()
print(f"Duplicates found: {duplicates}")
data = data.drop_duplicates(subset='review')
data = data.reset_index(drop=True)
print(f"New shape after removing duplicates: {data.shape}")
print(f"Null values: {data.isnull().sum().sum()}")

# ==================== TEXT PREPROCESSING FUNCTIONS ====================
def remove_punctuations(text):
    """Remove punctuation from text"""
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

# ==================== APPLY PREPROCESSING ====================
print("\n" + "="*80)
print("STEP 4: APPLYING TEXT PREPROCESSING")
print("="*80)

# Convert to lowercase to treat words case-insensitively
# 'Good' and 'good' should be the same feature
data["review"] = data["review"].apply(lambda x: " ".join(x.lower() for x in x.split()))
print("Lowercase conversion complete")

# Remove HTML links as they don't contribute to sentiment
data["review"] = data['review'].apply(
    lambda x: " ".join(re.sub(r'^https?:\/\/.*[\r\n]*', '', x, flags=re.MULTILINE) 
                      for x in x.split())
)
print("Links removed")

# Remove punctuation to focus on words only
data["review"] = data["review"].apply(remove_punctuations)
print("Punctuations removed")

# Remove numbers as they rarely indicate sentiment
data["review"] = data['review'].str.replace(r'\d+', '', regex=True)
print("Numbers removed")

# ==================== LOAD AND REMOVE STOPWORDS ====================
print("\n" + "="*80)
print("STEP 5: LOADING AND REMOVING STOPWORDS")
print("="*80)

# Use NLTK's English stopwords list
# Stopwords are common words that don't carry sentiment (the, a, is, etc.)

# Create stopwords directory structure
stopwords_dir = MODEL_DIR / "corpora" / "stopwords"
stopwords_dir.mkdir(parents=True, exist_ok=True)
stopwords_path = stopwords_dir / "english"

# Load stopwords from file or download if not exists
if stopwords_path.exists():
    with open(stopwords_path, 'r') as f:
        sw = f.read().splitlines()
    print(f"Loaded {len(sw)} stopwords from existing file")
else:
    print("Downloading stopwords...")
    nltk.download('stopwords', download_dir='../static/model', quiet=True)
    with open(stopwords_path, 'r') as f:
        sw = f.read().splitlines()
    print(f"Loaded {len(sw)} stopwords")

# Remove stopwords to reduce feature dimensionality
data['review'] = data['review'].apply(
    lambda x: " ".join(word for word in x.split() if word not in sw)
)
print("Stopwords removed")

# ==================== STEMMING ====================
print("\n" + "="*80)
print("STEP 6: REMOVE STEMMING")
print("="*80)

# Use Porter Stemmer to reduce words to root form
# 'running', 'runner', 'ran' all become 'run' - reduces vocabulary size
ps = PorterStemmer()
data['review'] = data['review'].apply(
    lambda x: " ".join(ps.stem(word) for word in x.split())
)
print("Stemming complete")

# Show sample of processed text
print("\nSample processed review:")
print(data['review'].iloc[0][:200] + "...")

# ==================== BUILD VOCABULARY ====================
print("\n" + "="*80)
print("STEP 7: BUILDING VOCABULARY")
print("="*80)

# Build vocabulary and filter by frequency
# Words appearing very rarely might be noise or typos
vocab = Counter()
for sentence in data['review']:
    vocab.update(sentence.split())

print(f"Total unique words: {len(vocab)}")

# Keep only words that appear more than 20 times
# This reduces dimensionality from 138k to 14k, removing noise and avoiding overfitting
tokens = [key for key in vocab if vocab[key] > 20]
print(f"Filtered tokens (frequency >20): {len(tokens)}")

# Save vocabulary for later use in prediction pipeline
def save_vocabulary(lines, filename):
    """Save vocabulary to file"""
    data = '\n'.join(lines)
    with open(filename, 'w', encoding="utf-8") as f:
        f.write(data)

vocab_path = MODEL_DIR / "vocabulary.txt"
save_vocabulary(tokens, str(vocab_path))
print(f"Vocabulary saved to: {vocab_path}")

# ==================== SPLIT DATA ====================
print("\n" + "="*80)
print("STEP 8: SPLITTING DATA INTO TRAIN/TEST SETS")
print("="*80)

# 80-20 train-test split
# Standard split that provides enough data for training and testing
X = data['review']
y = data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # random_state for reproducibility
)
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

print(f"\nClass distribution in training:")
print(y_train.value_counts())  # Verify balanced classes

# ==================== TF-IDF VECTORIZATION ====================
print("\n" + "="*80)
print("STEP 9: APPLYING TF-IDF VECTORIZATION")
print("="*80)

# Use TF-IDF instead of simple Bag-of-Words
# TF-IDF gives more weight to rare, important words and less to common words
tfidf_vectorizer = TfidfVectorizer(
    vocabulary=tokens,           # Use pre-built vocabulary for consistency
    lowercase=False,             # Already lowercased in preprocessing
    norm='l2',                   # L2 normalization for consistent vector lengths
    use_idf=True,                 # Use inverse document frequency weighting
    smooth_idf=True,              # Smooth idf weights to avoid division by zero
    sublinear_tf=False            
)

# Fit on training data only, transform both train and test
# Fit vectorizer ONLY on training data to prevent data leakage
X_train_vectorized = tfidf_vectorizer.fit_transform(X_train)
X_test_vectorized = tfidf_vectorizer.transform(X_test)

print(f"Training vectorized shape: {X_train_vectorized.shape}")
print(f"Test vectorized shape: {X_test_vectorized.shape}")

# Save vectorizer for prediction pipeline
vectorizer_path = MODEL_DIR / "tfidf_vectorizer.pickle"
with open(vectorizer_path, 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
print(f"Vectorizer saved to: {vectorizer_path}")

# ==================== TRAIN AND COMPARE MULTIPLE MODELS ====================
print("\n" + "="*80)
print("STEP 10: TRAINING AND COMPARING MULTIPLE MODELS")
print("="*80)

# Train multiple algorithms to find the best performer
# Different algorithms have different strengths for text classification
results = {}

# ----- 1. LOGISTIC REGRESSION (Base) -----
print("\n1. Training Logistic Regression (base)...")
print("-" * 60)

lr_base = LogisticRegression(random_state=42)
lr_base.fit(X_train_vectorized, y_train)
y_pred_lr = lr_base.predict(X_test_vectorized)
lr_accuracy = accuracy_score(y_test, y_pred_lr) * 100

results['Logistic Regression'] = {
    'model': lr_base,
    'accuracy': lr_accuracy,
    'predictions': y_pred_lr
}
print(f"Logistic Regression Accuracy: {lr_accuracy:.2f}%")

# ----- 2. RANDOM FOREST -----
print("\n2. Training Random Forest...")
print("-" * 60)

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
rf.fit(X_train_vectorized, y_train)
y_pred_rf = rf.predict(X_test_vectorized)
rf_accuracy = accuracy_score(y_test, y_pred_rf) * 100

results['Random Forest'] = {
    'model': rf,
    'accuracy': rf_accuracy,
    'predictions': y_pred_rf
}
print(f"Random Forest Accuracy: {rf_accuracy:.2f}%")

# ----- 3. NAIVE BAYES -----
print("\n3. Training Naive Bayes...")
print("-" * 60)

mnb = MultinomialNB()
mnb.fit(X_train_vectorized, y_train)
y_pred_mnb = mnb.predict(X_test_vectorized)
mnb_accuracy = accuracy_score(y_test, y_pred_mnb) * 100

results['Naive Bayes'] = {
    'model': mnb,
    'accuracy': mnb_accuracy,
    'predictions': y_pred_mnb
}
print(f"Naive Bayes Accuracy: {mnb_accuracy:.2f}%")

# ----- 4. DECISION TREE -----
print("\n4. Training Decision Tree...")
print("-" * 60)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_vectorized, y_train)
y_pred_dt = dt.predict(X_test_vectorized)
dt_accuracy = accuracy_score(y_test, y_pred_dt) * 100

results['Decision Tree'] = {
    'model': dt,
    'accuracy': dt_accuracy,
    'predictions': y_pred_dt
}
print(f"Decision Tree Accuracy: {dt_accuracy:.2f}%")

# ==================== SELECT BEST BASE MODEL ====================
print("\n" + "="*80)
print("STEP 11: SELECTING BEST BASE MODEL")
print("="*80)

print("\nModel Performance Summary:")
print("-" * 60)
print(f"{'Model':<20} {'Accuracy':>15}")
print("-" * 60)
for model_name, model_results in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
    print(f"{model_name:<20} {model_results['accuracy']:>14.2f}%")

# Use Logistic Regression as best model 
print("\n" + "-" * 60)
print("DECISION: Using Logistic Regression for hyperparameter tuning")
print("Reason: Based on notebook findings and literature, Logistic Regression")
print("provides the best balance of accuracy, interpretability, and speed")
print("for text classification with TF-IDF features.")

# ==================== HYPERPARAMETER TUNING ====================
print("\n" + "="*80)
print("STEP 12: HYPERPARAMETER TUNING WITH GRIDSEARCHCV")
print("="*80)

# GridSearchCV for systematic hyperparameter tuning
# Testing multiple values of C (regularization strength)
param_grid = {
    'C': [0.1, 1, 10],           # Regularization strength (lower = stronger regularization)
    'solver': ['liblinear'],       # Solver optimized for small datasets
    'max_iter': [100]              # Maximum iterations for convergence
}

print("\nHyperparameter grid:")
print(f"   C (regularization): {param_grid['C']}")
print(f"   solver: {param_grid['solver']}")
print(f"   max_iter: {param_grid['max_iter']}")
print(f"\n   Total combinations: {len(param_grid['C']) * len(param_grid['solver']) * len(param_grid['max_iter'])}")

# 3-fold cross-validation to prevent overfitting
print("\nRunning GridSearchCV with 3-fold cross-validation...")
print("This may take a few minutes...")

grid = GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid,
    cv=3,                          # 3-fold cross validation
    scoring='accuracy',            # Optimize for accuracy
    n_jobs=1,                      # Use 1 CPU core to avoid memory issues
    verbose=1                      # Show progress
)

grid.fit(X_train_vectorized, y_train)

print("\n" + "-" * 60)
print("GRID SEARCH RESULTS")
print("-" * 60)
print(f"Best parameters: {grid.best_params_}")
print(f"Best cross-validation accuracy: {grid.best_score_*100:.2f}%")

best_tuned_model = grid.best_estimator_

# ==================== EVALUATE TUNED MODEL ====================
print("\n" + "="*80)
print("STEP 13: EVALUATING TUNED MODEL ON TEST SET")
print("="*80)

y_pred_tuned = best_tuned_model.predict(X_test_vectorized)
tuned_accuracy = accuracy_score(y_test, y_pred_tuned) * 100

print(f"\nBase Logistic Regression Accuracy: {results['Logistic Regression']['accuracy']:.2f}%")
print(f"Tuned Logistic Regression Accuracy: {tuned_accuracy:.2f}%")
print(f"Improvement: {tuned_accuracy - results['Logistic Regression']['accuracy']:.2f}%")

print("\nClassification Report (Tuned Model):")
print(classification_report(y_test, y_pred_tuned))

print("\nConfusion Matrix (Tuned Model):")
print(confusion_matrix(y_test, y_pred_tuned))

# ==================== SAVE TUNED MODEL ====================
print("\n" + "="*80)
print("STEP 14: SAVING TUNED MODEL")
print("="*80)

# Save model with pickle for easy loading in FastAPI
model_path = MODEL_DIR / "model.pickle"
with open(model_path, 'wb') as f:
    pickle.dump(best_tuned_model, f)
print(f"Tuned model saved to: {model_path}")
print(f"Model: Logistic Regression (tuned)")
print(f"Best parameters: {grid.best_params_}")

# ==================== SAVE EVALUATION REPORT ====================
print("\n" + "="*80)
print("STEP 15: SAVING COMPREHENSIVE EVALUATION REPORT")
print("="*80)

# Save evaluation metrics for documentation and review
report_path = MODEL_DIR / "evaluation_report.txt"
with open(report_path, 'w') as f:
    f.write("="*70 + "\n")
    f.write("SENTIMENT ANALYSIS MODEL EVALUATION REPORT\n")
    f.write("="*70 + "\n\n")
    
    f.write("DATASET INFORMATION\n")
    f.write("-"*50 + "\n")
    f.write(f"Dataset: IMDb Movie Reviews\n")
    f.write(f"Total samples: {len(data)}\n")
    f.write(f"Training samples: {X_train.shape[0]}\n")
    f.write(f"Test samples: {X_test.shape[0]}\n")
    f.write(f"Vocabulary size: {len(tokens)}\n\n")
    
    f.write("PART 1: MODEL COMPARISON\n")
    f.write("-"*50 + "\n")
    for model_name, model_results in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        f.write(f"{model_name}: {model_results['accuracy']:.2f}%\n")
    f.write(f"\nSelected Best Model: Logistic Regression\n\n")
    
    f.write("PART 2: HYPERPARAMETER TUNING RESULTS\n")
    f.write("-"*50 + "\n")
    f.write(f"Best parameters: {grid.best_params_}\n")
    f.write(f"Best CV accuracy: {grid.best_score_*100:.2f}%\n\n")
    
    f.write("All combinations tested:\n")
    for i, params in enumerate(grid.cv_results_['params']):
        score = grid.cv_results_['mean_test_score'][i] * 100
        f.write(f"  {params}: {score:.2f}%\n")
    
    f.write("\nPART 3: FINAL MODEL PERFORMANCE\n")
    f.write("-"*50 + "\n")
    f.write(f"Base Logistic Regression Accuracy: {results['Logistic Regression']['accuracy']:.2f}%\n")
    f.write(f"Tuned Logistic Regression Accuracy: {tuned_accuracy:.2f}%\n")
    f.write(f"Improvement: {tuned_accuracy - results['Logistic Regression']['accuracy']:.2f}%\n\n")
    
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred_tuned))
    f.write("\nConfusion Matrix:\n")
    f.write(str(confusion_matrix(y_test, y_pred_tuned)))

print(f"Evaluation report saved to: {report_path}")

# ==================== VERIFY SAVED FILES ====================
print("\n" + "="*80)
print("STEP 16: VERIFYING SAVED FILES")
print("="*80)
print("-" * 60)

files_to_check = [
    ("model.pickle", MODEL_DIR / "model.pickle"),
    ("tfidf_vectorizer.pickle", MODEL_DIR / "tfidf_vectorizer.pickle"),
    ("vocabulary.txt", MODEL_DIR / "vocabulary.txt"),
    ("evaluation_report.txt", MODEL_DIR / "evaluation_report.txt"),
]

for name, path in files_to_check:
    if path.exists():
        size = path.stat().st_size / 1024
        print(f"{name}: {size:.1f} KB")
    else:
        print(f"{name}: NOT FOUND")

# ==================== SUMMARY ====================
print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print("\nFiles saved in ../static/model/:")
print(f"1. model.pickle - Logistic Regression (tuned with {grid.best_params_})")
print(f"2. tfidf_vectorizer.pickle - TF-IDF vectorizer")
print(f"3. vocabulary.txt - Vocabulary tokens ({len(tokens)} words)")
print(f"4. evaluation_report.txt - Complete evaluation report")

print("\nFinal Model Performance:")
print(f"   - Base Logistic Regression: {results['Logistic Regression']['accuracy']:.2f}%")
print(f"   - Tuned Logistic Regression: {tuned_accuracy:.2f}%")
print(f"   - Improvement: {tuned_accuracy - results['Logistic Regression']['accuracy']:.2f}%")
print(f"   - Best Parameters: {grid.best_params_}")
print("="*80)