import os
import pandas as pd
import numpy as np
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'archive', 'IMDB Dataset.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

print("--- Starting ML Pipeline ---")

# 1. Load Data
print(f"Loading data from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Loaded {len(df)} unique reviews.")

# 2. Preprocessing
print("Downloading NLTK resources...")
for res in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']:
    nltk.download(res, quiet=True)

STOP_WORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)

print("Preprocessing text (this may take a minute)...")
df['clean_review'] = df['review'].apply(preprocess_text)
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Save processed data
df[['clean_review', 'label']].to_csv(os.path.join(PROCESSED_DATA_DIR, 'processed_imdb_data.csv'), index=False)
print("Saved processed data.")

# 3. Training
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_review'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

print("Building and training pipeline (TF-IDF + Logistic Regression)...")
# Using Logistic Regression for stability and speed in this environment
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True)),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])

pipeline.fit(X_train, y_train)

# 4. Evaluation
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 5. Export
model_export_path = os.path.join(MODEL_DIR, 'best_model.joblib')
joblib.dump(pipeline, model_export_path)
print(f"Model exported to {model_export_path}")

print("--- Pipeline Completed Successfully ---")
