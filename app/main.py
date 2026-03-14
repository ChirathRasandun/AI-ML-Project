from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from app.schemas import PredictionRequest, PredictionResponse, HealthResponse,BatchPredictionRequest, BatchPredictionResponse
from app.model import SentimentModel

# Global model instance
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    # Load model on startup
    model = SentimentModel()
    yield
    # Cleanup on shutdown
    model = None

# Create FastAPI app
app = FastAPI(lifespan=lifespan)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(request: PredictionRequest):
    """Predict sentiment from text"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        sentiment, confidence = model.predict(request.text)
        return PredictionResponse(
            text=request.text,
            sentiment=sentiment,
            confidence=round(confidence, 4)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

    # ============= BATCH ENDPOINT (PART 4) =============

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict sentiment for multiple texts"""
    
    # Check model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Empty list check
    if not request.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    
    try:
        # Get predictions
        results = model.predict_batch(request.texts)
        
        # Format response
        predictions = [
            PredictionResponse(text=t, sentiment=s, confidence=round(c, 4))
            for t, s, c in results
        ]
        
        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))