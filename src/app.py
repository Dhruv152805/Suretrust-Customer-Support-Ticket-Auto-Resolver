"""
FastAPI deployment for the Customer Support Ticket Auto-Resolver.

Endpoints:
  POST /predict — classify a ticket and get suggested solutions

Usage:
    uvicorn src.app:app --reload
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional
from src.config import logger, BASE_DIR
import os

app = FastAPI(
    title="Customer Support Ticket Auto-Resolver",
    description=(
        "Automatically classify support tickets and suggest solutions "
        "using NLP + ML + Semantic Search with Ensemble Precision."
    ),
    version="1.1.0",
)

# Lazy-load pipeline to avoid slow startup on import
_pipeline = None


def get_pipeline(classifier_type: str = "ensemble"):
    """Get or create the pipeline singleton."""
    global _pipeline
    if _pipeline is None or _pipeline.classifier_type != classifier_type:
        from src.pipeline import SupportPipeline
        _pipeline = SupportPipeline(classifier_type=classifier_type)
    return _pipeline


# ─── Request / Response Models ───────────────────────────────────────────────

class TicketRequest(BaseModel):
    """Input: a customer complaint text."""
    text: str = Field(..., description="The customer's complaint or query text.")
    model: Optional[str] = Field(
        default="ensemble",
        description="Model to use: 'tfidf_lr', 'w2v_xgb', 'bert', or 'ensemble'."
    )
    top_k: Optional[int] = Field(
        default=5,
        description="Number of similar tickets to retrieve."
    )

    model_config = {"json_schema_extra": {
        "examples": [{
            "text": "Where is my order? I've been waiting 2 weeks!",
            "model": "ensemble",
            "top_k": 5
        }]
    }}


class SuggestionResponse(BaseModel):
    """A single similar ticket with its solution."""
    similar_issue: str
    suggested_solution: str
    category: str
    similarity_score: float
    rerank_score: Optional[float] = None


class FeedbackRequest(BaseModel):
    """User feedback for a prediction."""
    query: str
    predicted_category: str
    is_correct: bool
    suggested_solution_id: Optional[int] = None
    comments: Optional[str] = None


class PredictionResponse(BaseModel):
    """Full prediction output."""
    query: str
    predicted_category: str
    confidence: float
    model_used: str
    suggested_solutions: list[SuggestionResponse]
    llm_response: Optional[str] = None


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/", response_class=FileResponse)
def read_index():
    """Serve the frontend dashboard."""
    index_path = os.path.join(BASE_DIR, "src", "static", "index.html")
    if not os.path.exists(index_path):
        # Fallback if index.html hasn't been created yet
        return {"status": "running", "message": "Frontend index.html not found yet. Please wait."}
    return FileResponse(index_path)


@app.get("/health")
def root():
    """Health check."""
    return {
        "status": "running",
        "service": "Customer Support Ticket Auto-Resolver (10/10 Polish)",
        "docs": "/docs"
    }


# Mount static files (css, js, images)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "src", "static")), name="static")


@app.post("/predict", response_model=PredictionResponse)
def predict_ticket(request: TicketRequest):
    """
    Classify a customer support ticket and suggest solutions.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    valid_models = ["tfidf_lr", "w2v_xgb", "bert", "ensemble"]
    if request.model not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Choose from: {valid_models}"
        )

    try:
        logger.info(f"Predicting with model: {request.model}")
        pipeline = get_pipeline(request.model)
        result = pipeline.predict(
            request.text, 
            top_k=request.top_k or 5,
            rerank=True # Always rerank for 10/10 quality
        )
        return result
    except Exception as e:
        import traceback
        logger.error(f"Prediction failed: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
def submit_feedback(feedback: FeedbackRequest):
    """
    Submit feedback for a prediction (Active Learning Loop).
    """
    import os
    import json
    from src.config import MODELS_DIR
    
    feedback_file = os.path.join(MODELS_DIR, "feedback_log.json")
    
    try:
        data = []
        if os.path.exists(feedback_file):
            with open(feedback_file, "r") as f:
                data = json.load(f)
        
        from datetime import datetime
        feedback_entry = feedback.model_dump()
        feedback_entry["timestamp"] = datetime.now().isoformat()
        data.append(feedback_entry)
        
        with open(feedback_file, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Feedback received for query: {feedback.query[:50]}...")
        return {"status": "success", "message": "Feedback received. Thank you!"}
    except Exception as e:
        logger.error(f"Feedback submission failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
def list_models():
    """List available models and their status."""
    import os
    from src.config import MODELS_DIR
    return {
        "available_models": {
            "tfidf_lr": os.path.exists(os.path.join(MODELS_DIR, "tfidf_lr_model.pkl")),
            "w2v_xgb": os.path.exists(os.path.join(MODELS_DIR, "xgb_model.pkl")),
            "bert": os.path.exists(os.path.join(MODELS_DIR, "bert_model")),
            "ensemble": os.path.exists(os.path.join(MODELS_DIR, "bert_model")) and os.path.exists(os.path.join(MODELS_DIR, "tfidf_lr_model.pkl"))
        },
        "faiss_index": os.path.exists(os.path.join(MODELS_DIR, "faiss_index.bin")),
    }
