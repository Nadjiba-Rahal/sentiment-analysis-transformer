from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(title="Sentiment Analysis API")

# Load model once (IMPORTANT: production optimization)
MODEL_PATH = "outputs/best_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()


class TextRequest(BaseModel):
    text: str


def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    label = torch.argmax(probs).item()
    confidence = torch.max(probs).item()

    return {
        "sentiment": "positive" if label == 1 else "negative",
        "confidence": float(confidence)
    }


@app.post("/predict")
def get_prediction(req: TextRequest):
    return predict(req.text)


@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running "}