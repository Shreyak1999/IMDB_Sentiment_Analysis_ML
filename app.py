from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.predict import predict_sentiment

app = FastAPI(title="Text Sentiment Analyzer")

# Define request schema
class TextRequest(BaseModel):
    text: str

# Define response schema
class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float

@app.post("/analyze", response_model=SentimentResponse)
def analyze_sentiment(request: TextRequest):
    if request.text.strip() == "":
        raise HTTPException(status_code=400, detail="Please provide some text.")
    
    label, prob = predict_sentiment(request.text)
    return SentimentResponse(sentiment=label, confidence=prob)

# Optional: simple health check
@app.get("/health")
def health_check():
    return {"status": "ok"}
