"""
app/preprocessing.py

Centralised text preprocessing module.
This mirrors the exact pipeline used in Notebook 01 so that training
and inference transformations are always in sync.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ── Download NLTK assets on first import ─────────────────────────────────────
for _resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(_resource, quiet=True)

_STOP_WORDS  = set(stopwords.words("english"))
_lemmatizer  = WordNetLemmatizer()


def preprocess(text: str) -> str:
    """
    Full NLP pipeline applied identically during training and serving.

    Steps
    -----
    1. Lowercase
    2. Strip HTML tags
    3. Remove URLs
    4. Keep only alphabetic characters
    5. Collapse whitespace
    6. Tokenise
    7. Stopword removal + WordNet lemmatisation
    """
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [
        _lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in _STOP_WORDS and len(tok) > 2
    ]
    return " ".join(tokens)
