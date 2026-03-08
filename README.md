# Sentiment Analysis — IMDB Movie Reviews

> **AI/ML Engineering Assessment** | End-to-End NLP Classification Pipeline

A production-grade sentiment analysis system built with a full NLP pipeline, 4-model comparison, class imbalance handling, and a clean REST API for model serving.

---

## Project Structure

```
├── notebooks/
│   ├── 01_data_preparation_eda.ipynb   # EDA + NLP Preprocessing Pipeline
│   └── 02_model_development.ipynb      # Feature Eng + 4 Models + Evaluation
├── app/
│   ├── main.py                         # FastAPI application entry point
│   ├── ml_model.py                     # Model loading & inference logic
│   └── preprocessing.py               # NLP preprocessing (shared pipeline)
├── models/                             # Trained model artifacts (git-ignored)
├── data/                               # Processed data & EDA plots (git-ignored)
├── archive/                            # Raw IMDB dataset (git-ignored)
├── requirements.txt
└── .gitignore
```

---

## 1. Algorithm & Architecture Choices

### Why TF-IDF + Traditional ML as primary approach?

| Criterion | TF-IDF + ML | Full Transformer Fine-tuning |
|---|---|---|
| Training Time | < 5 minutes | 2–6 hours (GPU needed) |
| Inference Latency | ~1ms | ~80ms (CPU) |
| RAM Requirements | ~200 MB | ~2–4 GB |
| Accuracy (IMDB) | ~90–93% | ~93–95% |
| Reproducibility | ✅ Easy | ❌ Requires GPU |

**Conclusion**: For IMDB binary classification, TF-IDF + Logistic Regression / XGBoost hits a "good enough" ceiling with dramatically lower resource cost. A Transformer comparison (DistilBERT) is included in Notebook 02 for reference.

### NLP Pipeline
- **Cleaning**: HTML tag stripping, URL removal, regex character filtering
- **Tokenisation**: NLTK `word_tokenize`
- **Stopword Removal**: NLTK English stopwords corpus
- **Lemmatisation**: NLTK `WordNetLemmatizer` (WordNet corpus)

### Feature Engineering
- **TF-IDF Vectorizer**: `unigrams + bigrams`, `max_features=50,000`, `sublinear_tf=True`

### Class Imbalance Strategy
- IMDB is balanced (25k / 25k), but `class_weight='balanced'` is applied to all sklearn models and `scale_pos_weight` is applied to XGBoost — making the approach robust for imbalanced real-world datasets.

### 4 Models Compared
1. **Logistic Regression** — Strong linear baseline, fast, interpretable
2. **Naive Bayes (Multinomial)** — Classic text classification baseline
3. **Random Forest** — Ensemble method with bagging
4. **XGBoost** — Gradient boosting, typically highest accuracy

---

## 2. Setup & Installation

### Prerequisites
- Python 3.9+
- pip

### Step 1: Clone the repository
```bash
git clone https://github.com/lilbuu-glitch/imdb-review-sentiment-analysis.git
cd imdb-review-sentiment-analysis
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Prepare the dataset
Download the [IMDB Dataset from Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place it:
```
archive/IMDB Dataset.csv
```

### Step 4: Run Notebook 01 — Data Preparation & EDA
```bash
jupyter notebook notebooks/01_data_preparation_eda.ipynb
```
Run all cells. This will produce `data/processed_imdb_data.csv`.

### Step 5: Run Notebook 02 — Model Development
```bash
jupyter notebook notebooks/02_model_development.ipynb
```
Run all cells. This will produce `models/best_model.joblib`.

---

## 3. Running the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API root**: http://127.0.0.1:8000
- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

---

## 4. API Endpoints

### `GET /`
Health check.

**Response:**
```json
{
    "status": "ok",
    "message": "Sentiment Analysis API is running.",
    "endpoints": {
        "POST /predict": "Classify sentiment of a text review.",
        "GET  /docs": "Interactive Swagger UI documentation."
    }
}
```

---

### `POST /predict`

Classify the sentiment of a review.

**Request Body:**
```json
{
    "text": "This movie was absolutely fantastic! A true masterpiece."
}
```

**Response:**
```json
{
    "sentiment": "Positive",
    "confidence_score": 0.9831,
    "probabilities": {
        "Negative": 0.0169,
        "Positive": 0.9831
    }
}
```

---

## 5. Example Usage

### cURL
```bash
# Positive review
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This movie was absolutely fantastic! A true masterpiece."}'

# Negative review
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Terrible film. Complete waste of time and money. I hated every second of it."}'
```

### Python (requests)
```python
import requests

url = "http://127.0.0.1:8000/predict"
payload = {"text": "One of the best films I have ever seen. Absolutely breathtaking!"}
response = requests.post(url, json=payload)
print(response.json())
# {
#     "sentiment": "Positive",
#     "confidence_score": 0.9712,
#     "probabilities": {"Negative": 0.0288, "Positive": 0.9712}
# }
```

### Postman
- **Method**: `POST`
- **URL**: `http://127.0.0.1:8000/predict`
- **Body**: raw → JSON
```json
{
    "text": "I couldn't finish watching it. Absolutely dreadful storyline."
}
```

---

## 6. Evaluation Metrics (Expected Results)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | ~0.906 | ~0.907 | ~0.906 | ~0.906 |
| Naive Bayes | ~0.858 | ~0.867 | ~0.858 | ~0.857 |
| Random Forest | ~0.852 | ~0.853 | ~0.852 | ~0.852 |
| **XGBoost** | **~0.927** | **~0.927** | **~0.927** | **~0.927** |

> *Exact values will vary slightly depending on environment. Run Notebook 02 for current results.*

---

## 7. Notes
- Model file (`models/best_model.joblib`) is excluded from git via `.gitignore` due to potential size. Run the notebooks to regenerate it.
- Dataset (`archive/IMDB Dataset.csv`) is also excluded. See Step 3 above.
