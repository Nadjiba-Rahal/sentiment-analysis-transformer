from fastapi import FastAPI
from pydantic import BaseModel, Field
from predict import SentimentPredictor

app = FastAPI(title="Sentiment Analysis API")

predictor = SentimentPredictor()


class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


def predict(text: str):
    result = predictor.predict_one(text)

    return {
        "sentiment": result["label"].split()[0].lower(),
        "confidence": result["confidence"],
        "probabilities": result["proba"],
    }


@app.post("/predict")
def get_prediction(req: TextRequest):
    return predict(req.text)


@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model": "distilbert-sentiment", "device": str(predictor.device)}