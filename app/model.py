import pickle
import pandas as pd
import re
import string
from pathlib import Path
from nltk.stem import PorterStemmer
import nltk

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class SentimentModel:
    def __init__(self):
        # Set paths
        self.model_dir = Path(__file__).parent.parent / "static" / "model"
        self.ps = PorterStemmer()
        
        # Load artifacts
        self.load_artifacts()
    
    def load_artifacts(self):
        # Load model
        with open(self.model_dir / "model.pickle", 'rb') as f:
            self.model = pickle.load(f)
        
        # Load vectorizer
        with open(self.model_dir / "tfidf_vectorizer.pickle", 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        # Load stopwords
        with open(self.model_dir / "corpora" / "stopwords" / "english", 'r') as f:
            self.stopwords = f.read().splitlines()
    
    def remove_punctuations(self, text):
        for punctuation in string.punctuation:
            text = text.replace(punctuation, '')
        return text
    
    def preprocess(self, text):
        # Create DataFrame (matching your training code)
        data = pd.DataFrame([text], columns=['review'])
        
        # Lowercase
        data["review"] = data["review"].apply(lambda x: " ".join(x.lower() for x in x.split()))
        
        # Remove links
        data["review"] = data['review'].apply(
            lambda x: " ".join(re.sub(r'^https?:\/\/.*[\r\n]*', '', x, flags=re.MULTILINE) 
                              for x in x.split())
        )
        
        # Remove punctuations
        data["review"] = data["review"].apply(self.remove_punctuations)
        
        # Remove numbers
        data["review"] = data['review'].str.replace(r'\d+', '', regex=True)
        
        # Remove stopwords
        data["review"] = data["review"].apply(
            lambda x: " ".join(word for word in x.split() if word not in self.stopwords)
        )
        
        # Stemming
        data["review"] = data["review"].apply(
            lambda x: " ".join(self.ps.stem(word) for word in x.split())
        )
        
        return data["review"].iloc[0]
    
    def predict(self, text):
        # Preprocess
        processed = self.preprocess(text)
        
        # Vectorize
        vectorized = self.vectorizer.transform([processed])
        
        # Predict
        prediction = self.model.predict(vectorized)[0]
        
        # Get confidence
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(vectorized)[0]
            class_index = list(self.model.classes_).index(prediction)
            confidence = float(probabilities[class_index])
        else:
            confidence = 0.0
        
        return prediction, confidence
    
    # Add BATCH METHOD
    def predict_batch(self, texts: list) -> list:
        """Predict sentiment for multiple texts"""
        results = []
        
        for text in texts:
            try:
                sentiment, confidence = self.predict(text)
                results.append((text, sentiment, confidence))
            except Exception as e:
                results.append((text, "error", 0.0))
        
        return results