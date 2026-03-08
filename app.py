"""
🎬 IMDB Sentiment Analysis API
FastAPI application for sentiment classification of movie reviews.

Features:
- Multiple model support (Logistic Regression, SVM, Random Forest, BERT)
- Probability outputs with confidence scores
- Input validation and preprocessing
- Comprehensive error handling
- Swagger documentation
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import re
import joblib
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging
from datetime import datetime
import asyncio
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="IMDB Sentiment Analysis API",
    description="Advanced sentiment analysis API for movie reviews using multiple ML models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Download NLTK resources (only once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    logger.info("Downloading NLTK resources...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

# Pydantic models for request/response
class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Movie review text to analyze")
    model: str = Field(default="logistic_regression", 
                      description="Model to use for prediction",
                      regex="^(logistic_regression|svm|random_forest|bert)$")

    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()

class SentimentResponse(BaseModel):
    sentiment: str = Field(..., description="Predicted sentiment (positive/negative)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    probability: Dict[str, float] = Field(..., description="Probability distribution")
    model_used: str = Field(..., description="Model used for prediction")
    processing_time: float = Field(..., description="Processing time in seconds")

class BatchReviewRequest(BaseModel):
    reviews: List[str] = Field(..., min_items=1, max_items=100, description="List of review texts")
    model: str = Field(default="logistic_regression", 
                      regex="^(logistic_regression|svm|random_forest|bert)$")

class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    timestamp: str

class ModelInfo(BaseModel):
    name: str
    type: str
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    description: str

# Text preprocessing class
class TextPreprocessor:
    """Comprehensive text preprocessing for sentiment analysis."""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Keep important sentiment words
        self.stop_words.discard('not')
        self.stop_words.discard('no')
        self.stop_words.discard('nor')
        
    def clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        # Remove special characters and numbers (keep letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_remove_stopwords(self, text: str) -> List[str]:
        """Tokenize and remove stopwords."""
        tokens = word_tokenize(text)
        # Remove stopwords and short words
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        return tokens
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens."""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text: str) -> str:
        """Full preprocessing pipeline."""
        # Step 1: Clean text
        cleaned = self.clean_text(text)
        
        # Step 2: Tokenize and remove stopwords
        tokens = self.tokenize_and_remove_stopwords(cleaned)
        
        # Step 3: Lemmatize
        lemmatized = self.lemmatize_tokens(tokens)
        
        # Join back to string
        return ' '.join(lemmatized)

# Model manager class
class ModelManager:
    """Manage loading and inference for different models."""
    
    def __init__(self):
        self.models = {}
        self.preprocessor = TextPreprocessor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_info = {
            'logistic_regression': {
                'type': 'traditional_ml',
                'description': 'Fast and interpretable linear model with TF-IDF features'
            },
            'svm': {
                'type': 'traditional_ml', 
                'description': 'Support Vector Machine with linear kernel for robust classification'
            },
            'random_forest': {
                'type': 'ensemble',
                'description': 'Random Forest ensemble method for non-linear patterns'
            },
            'bert': {
                'type': 'deep_learning',
                'description': 'Fine-tuned BERT transformer for contextual understanding'
            }
        }
        
    def load_models(self):
        """Load all available models."""
        logger.info("Loading models...")
        
        # Load traditional ML models
        model_files = {
            'logistic_regression': 'logistic_regression_model.joblib',
            'svm': 'svm_model.joblib',
            'random_forest': 'random_forest_model.joblib'
        }
        
        for model_name, filename in model_files.items():
            try:
                if os.path.exists(filename):
                    self.models[model_name] = joblib.load(filename)
                    logger.info(f"✅ Loaded {model_name} model")
                else:
                    logger.warning(f"⚠️ {filename} not found")
            except Exception as e:
                logger.error(f"❌ Error loading {model_name}: {str(e)}")
        
        # Load BERT model
        try:
            if os.path.exists('bert_sentiment_model'):
                self.models['bert'] = AutoModelForSequenceClassification.from_pretrained('bert_sentiment_model')
                self.bert_tokenizer = AutoTokenizer.from_pretrained('bert_sentiment_model')
                self.models['bert'].to(self.device)
                self.models['bert'].eval()
                logger.info("✅ Loaded BERT model")
            else:
                logger.warning("⚠️ BERT model directory not found")
        except Exception as e:
            logger.error(f"❌ Error loading BERT model: {str(e)}")
        
        # Load model comparison results
        try:
            if os.path.exists('model_comparison_results.csv'):
                comparison_df = pd.read_csv('model_comparison_results.csv', index_col=0)
                for model_name in self.model_info.keys():
                    if model_name in comparison_df.index:
                        self.model_info[model_name]['accuracy'] = comparison_df.loc[model_name, 'accuracy']
                        self.model_info[model_name]['f1_score'] = comparison_df.loc[model_name, 'f1_score']
        except Exception as e:
            logger.error(f"❌ Error loading model comparison results: {str(e)}")
        
        logger.info(f"Models loaded: {list(self.models.keys())}")
    
    def predict_traditional_ml(self, text: str, model_name: str) -> tuple:
        """Predict using traditional ML models."""
        model = self.models[model_name]
        
        # Preprocess text
        processed_text = self.preprocessor.preprocess(text)
        
        # Make prediction
        prediction = model.predict([processed_text])[0]
        probabilities = model.predict_proba([processed_text])[0]
        
        return prediction, probabilities
    
    def predict_bert(self, text: str) -> tuple:
        """Predict using BERT model."""
        model = self.models['bert']
        tokenizer = self.bert_tokenizer
        
        # Tokenize input
        inputs = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt'
        )
        
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
            prediction = np.argmax(probabilities)
        
        return prediction, probabilities
    
    def predict(self, text: str, model_name: str) -> tuple:
        """Main prediction method."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        if model_name == 'bert':
            return self.predict_bert(text)
        else:
            return self.predict_traditional_ml(text, model_name)

# Initialize model manager
model_manager = ModelManager()

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the application."""
    logger.info("🚀 Starting IMDB Sentiment Analysis API...")
    model_manager.load_models()
    logger.info("✅ API startup complete!")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and available models."""
    return HealthResponse(
        status="healthy",
        models_loaded=list(model_manager.models.keys()),
        timestamp=datetime.now().isoformat()
    )

# Model information endpoint
@app.get("/models", response_model=List[ModelInfo])
async def get_models():
    """Get information about available models."""
    models_info = []
    for name, info in model_manager.model_info.items():
        if name in model_manager.models:
            models_info.append(ModelInfo(
                name=name,
                type=info['type'],
                accuracy=info.get('accuracy'),
                f1_score=info.get('f1_score'),
                description=info['description']
            ))
    return models_info

# Single prediction endpoint
@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: ReviewRequest):
    """Predict sentiment for a single review."""
    import time
    start_time = time.time()
    
    try:
        # Validate model availability
        if request.model not in model_manager.models:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' not available. Available models: {list(model_manager.models.keys())}"
            )
        
        # Make prediction
        prediction, probabilities = model_manager.predict(request.text, request.model)
        
        # Convert prediction to sentiment label
        sentiment = "positive" if prediction == 1 else "negative"
        
        # Calculate confidence
        confidence = float(max(probabilities))
        
        # Create probability dictionary
        prob_dict = {
            "negative": float(probabilities[0]),
            "positive": float(probabilities[1])
        }
        
        processing_time = time.time() - start_time
        
        return SentimentResponse(
            sentiment=sentiment,
            confidence=confidence,
            probability=prob_dict,
            model_used=request.model,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Batch prediction endpoint
@app.post("/predict/batch", response_model=List[SentimentResponse])
async def predict_batch(request: BatchReviewRequest):
    """Predict sentiment for multiple reviews."""
    import time
    start_time = time.time()
    
    try:
        # Validate model availability
        if request.model not in model_manager.models:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' not available. Available models: {list(model_manager.models.keys())}"
            )
        
        # Validate batch size
        if len(request.reviews) > 100:
            raise HTTPException(status_code=400, detail="Maximum batch size is 100 reviews")
        
        results = []
        
        for text in request.reviews:
            try:
                # Make prediction
                prediction, probabilities = model_manager.predict(text, request.model)
                
                # Convert prediction to sentiment label
                sentiment = "positive" if prediction == 1 else "negative"
                
                # Calculate confidence
                confidence = float(max(probabilities))
                
                # Create probability dictionary
                prob_dict = {
                    "negative": float(probabilities[0]),
                    "positive": float(probabilities[1])
                }
                
                results.append(SentimentResponse(
                    sentiment=sentiment,
                    confidence=confidence,
                    probability=prob_dict,
                    model_used=request.model,
                    processing_time=0.0  # Individual timing not needed for batch
                ))
                
            except Exception as e:
                logger.error(f"Error processing review: {str(e)}")
                # Add error response for this review
                results.append(SentimentResponse(
                    sentiment="error",
                    confidence=0.0,
                    probability={"negative": 0.0, "positive": 0.0},
                    model_used=request.model,
                    processing_time=0.0
                ))
        
        processing_time = time.time() - start_time
        logger.info(f"Batch prediction completed in {processing_time:.2f}s for {len(request.reviews)} reviews")
        
        return results
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "🎬 IMDB Sentiment Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "models": "/models",
        "endpoints": {
            "predict": "/predict - Single review prediction",
            "predict_batch": "/predict/batch - Batch prediction",
            "health": "/health - Health check",
            "models": "/models - Available models"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
