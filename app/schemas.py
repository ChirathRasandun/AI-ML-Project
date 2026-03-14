from pydantic import BaseModel, validator
from typing import List

# ============= SINGLE PREDICTION MODELS =============

class PredictionRequest(BaseModel):
    text: str
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()

class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float

class HealthResponse(BaseModel):
    status: str


# ============= BATCH PREDICTION MODELS (PART 4) =============

class BatchPredictionRequest(BaseModel):
    texts: List[str]
    
    @validator('texts')
    def validate_texts(cls, v):
        # Empty list check (required for bonus)
        if not v:
            raise ValueError('Texts list cannot be empty')
        return v

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int