"""
app/main.py

FastAPI application entry point.

Endpoints
---------
GET  /          — Health check / API info
POST /predict   — Sentiment prediction with confidence score
GET  /docs      — Swagger UI (auto-generated)
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from app.ml_model import predict

# ── Application metadata ──────────────────────────────────────────────────────
app = FastAPI(
    title="Sentiment Analysis API",
    description=(
        "REST API for classifying the sentiment of movie/product reviews. "
        "Returns a sentiment label and confidence score."
    ),
    version="1.0.0",
    contact={"name": "AI Engineer Assessment"},
)


# ── Request / Response schemas ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=3,
        max_length=10_000,
        examples=["This movie was absolutely fantastic! A true masterpiece."],
        description="Raw review text to classify.",
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("'text' must not be blank or whitespace only.")
        return v


class PredictResponse(BaseModel):
    sentiment: str = Field(..., examples=["Positive"])
    confidence_score: float = Field(..., ge=0.0, le=1.0, examples=[0.9423])
    probabilities: dict = Field(
        ..., examples=[{"Negative": 0.0577, "Positive": 0.9423}]
    )


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    """Returns API status and available endpoints."""
    return {
        "status":    "ok",
        "message":   "Sentiment Analysis API is running.",
        "endpoints": {
            "POST /predict": "Classify sentiment of a text review.",
            "GET  /docs":    "Interactive Swagger UI documentation.",
        },
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict_sentiment(request: PredictRequest):
    """
    **Classify the sentiment** of a given text review.

    - Returns **sentiment** label: `Positive` or `Negative`
    - Returns **confidence_score**: probability of the predicted class
    - Returns **probabilities**: full class probability distribution
    """
    try:
        result = predict(request.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    return JSONResponse(content=result)
