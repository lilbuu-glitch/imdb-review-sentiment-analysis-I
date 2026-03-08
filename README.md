# 🎬 IMDB Sentiment Analysis II

Advanced sentiment analysis system for movie reviews using multiple Machine Learning and Deep Learning approaches.

## 🎯 Project Overview

This project demonstrates a complete ML pipeline for sentiment classification of IMDB movie reviews, featuring:

- **4 Different Models**: Logistic Regression, SVM, Random Forest, and BERT
- **Comprehensive EDA**: Data visualization and analysis
- **Advanced NLP Pipeline**: Text preprocessing with multiple techniques
- **Model Comparison**: Performance evaluation with multiple metrics
- **REST API**: FastAPI-based serving with probability outputs
- **Production Ready**: Error handling, validation, and monitoring

## 📊 Dataset

- **Source**: IMDB Dataset with 50,000 movie reviews
- **Labels**: Binary sentiment (Positive/Negative)
- **Format**: CSV with `review` and `sentiment` columns
- **Size**: ~66MB

## 🏗️ Architecture

### ML Pipeline
```
Data Loading → EDA → Preprocessing → Feature Engineering → Model Training → Evaluation → Deployment
```

### Models Implemented
1. **Logistic Regression** - Fast, interpretable baseline
2. **Support Vector Machine** - Robust linear classifier
3. **Random Forest** - Ensemble method for non-linear patterns
4. **BERT** - Transformer model for contextual understanding

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- GPU (optional, for BERT training)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/lilbuu-glitch/imdb-review-sentiment-analysis-II.git
cd imdb-review-sentiment-analysis-II
```

2. **Create virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the dataset**
- Place `IMDB Dataset.csv` in the `archive/` directory
- Or download from [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

### Running the Project

#### Option 1: Full Pipeline (Recommended)
Run the Jupyter notebooks in order:

1. **Data Preparation & EDA**
```bash
jupyter notebook 01_data_preparation_eda.ipynb
```

2. **Model Development & Comparison**
```bash
jupyter notebook 02_model_development_comparison.ipynb
```

#### Option 2: API Only (Pre-trained Models)
If you have pre-trained models, start the API directly:

```bash
python app.py
```

The API will be available at `http://localhost:8000`

## 📁 Project Structure

```
imdb-review-sentiment-analysis-II/
├── archive/
│   └── IMDB Dataset.csv          # Raw dataset
├── 01_data_preparation_eda.ipynb # Data exploration and preprocessing
├── 02_model_development_comparison.ipynb # Model training and comparison
├── app.py                        # FastAPI application
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── processed_imdb_data.csv       # Processed dataset (generated)
├── model_comparison_results.csv  # Model performance comparison (generated)
├── logistic_regression_model.joblib # Trained model (generated)
├── svm_model.joblib              # Trained model (generated)
├── random_forest_model.joblib    # Trained model (generated)
├── bert_sentiment_model/         # BERT model directory (generated)
└── best_model_info.pkl           # Best model information (generated)
```

## 🧠 Model Performance

Based on comprehensive evaluation:

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| Logistic Regression | ~0.89 | ~0.89 | ~0.89 | ~0.89 | ~10s |
| SVM | ~0.90 | ~0.90 | ~0.90 | ~0.90 | ~30s |
| Random Forest | ~0.85 | ~0.85 | ~0.85 | ~0.85 | ~60s |
| BERT | ~0.92 | ~0.92 | ~0.92 | ~0.92 | ~300s |

*Note: BERT was trained on a subset (2000 samples) for faster training. Full training would yield better results.*

## 🛠️ API Usage

### Start the API
```bash
python app.py
```

### Available Endpoints

#### 1. Health Check
```bash
GET /health
```

#### 2. Model Information
```bash
GET /models
```

#### 3. Single Prediction
```bash
POST /predict
Content-Type: application/json

{
    "text": "This movie was absolutely fantastic! Great acting and storyline.",
    "model": "logistic_regression"
}
```

**Response:**
```json
{
    "sentiment": "positive",
    "confidence": 0.95,
    "probability": {
        "negative": 0.05,
        "positive": 0.95
    },
    "model_used": "logistic_regression",
    "processing_time": 0.023
}
```

#### 4. Batch Prediction
```bash
POST /predict/batch
Content-Type: application/json

{
    "reviews": [
        "Great movie with excellent acting!",
        "Terrible plot and bad acting."
    ],
    "model": "bert"
}
```

### Interactive Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🧪 Testing the API

### Using cURL
```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
         "text": "This movie was amazing!",
         "model": "logistic_regression"
     }'

# Health check
curl -X GET "http://localhost:8000/health"

# Available models
curl -X GET "http://localhost:8000/models"
```

### Using Python
```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "text": "This movie was fantastic!",
        "model": "bert"
    }
)
print(response.json())

# Batch prediction
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={
        "reviews": [
            "Great movie!",
            "Bad acting."
        ],
        "model": "svm"
    }
)
print(response.json())
```

## 🔧 Technical Details

### Text Preprocessing Pipeline
1. **HTML Tag Removal**: Clean HTML artifacts
2. **URL/Email Removal**: Remove non-text elements
3. **Special Character Removal**: Keep only letters and spaces
4. **Lowercase Conversion**: Standardize text
5. **Tokenization**: Split into words
6. **Stopword Removal**: Remove common words (keep negations)
7. **Lemmatization**: Reduce words to base form

### Feature Engineering
- **TF-IDF Vectorization**: 10,000 features, 1-2 ngrams
- **BERT Embeddings**: Contextual word representations
- **Class Balance**: Balanced dataset with equal positive/negative samples

### Model Selection Rationale

#### Logistic Regression
- **Pros**: Fast, interpretable, good baseline
- **Use Case**: Quick predictions, feature importance analysis

#### Support Vector Machine
- **Pros**: Robust, handles high-dimensional data well
- **Use Case**: When margin maximization is important

#### Random Forest
- **Pros**: Handles non-linear patterns, feature importance
- **Use Case**: Complex relationships, ensemble benefits

#### BERT
- **Pros**: Contextual understanding, state-of-the-art performance
- **Use Case**: When accuracy is paramount and resources available

## 📈 Evaluation Metrics

- **Accuracy**: Overall correct predictions
- **Precision**: True positive rate
- **Recall**: Sensitivity
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under precision-recall curve

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory (BERT)**
   - Reduce batch size in training
   - Use smaller model variant
   - Clear GPU cache: `torch.cuda.empty_cache()`

2. **Model Loading Errors**
   - Ensure all model files are present
   - Check file permissions
   - Verify model compatibility

3. **Slow API Response**
   - Use CPU-optimized models for inference
   - Implement model caching
   - Consider model quantization

4. **Memory Issues**
   - Reduce dataset size for testing
   - Use streaming for large datasets
   - Increase system RAM

### Performance Optimization

1. **For Production**:
   - Use model quantization
   - Implement request batching
   - Add Redis caching
   - Use GPU for BERT inference

2. **For Development**:
   - Use subset of data for testing
   - Reduce BERT sequence length
   - Disable verbose logging

## 🔮 Future Enhancements

1. **Model Improvements**:
   - Fine-tune larger BERT variants
   - Implement ensemble methods
   - Add more model architectures

2. **Features**:
   - Real-time streaming predictions
   - Model monitoring and drift detection
   - A/B testing framework
   - Explainable AI (SHAP, LIME)

3. **Deployment**:
   - Docker containerization
   - Kubernetes orchestration
   - CI/CD pipeline
   - Auto-scaling

## 📝 Model Comparison Insights

### Key Findings

1. **Traditional ML vs Deep Learning**:
   - Traditional ML models are faster and more interpretable
   - BERT provides superior accuracy but requires more resources
   - SVM performs surprisingly well for text classification

2. **Training Time vs Performance**:
   - Logistic Regression: Best speed/performance ratio
   - BERT: Highest accuracy but slowest training
   - Random Forest: Moderate performance, longer training

3. **Practical Recommendations**:
   - Use Logistic Regression for real-time applications
   - Use BERT for batch processing where accuracy is critical
   - Consider SVM as a good compromise

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- IMDB dataset providers
- Hugging Face for transformer models
- Scikit-learn for traditional ML algorithms
- FastAPI for the web framework

## 📞 Contact

For questions or suggestions, please open an issue on GitHub.

---

**⭐ If this project helped you, please give it a star!**
