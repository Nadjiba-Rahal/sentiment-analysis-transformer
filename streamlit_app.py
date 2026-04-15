import streamlit as st
from predict import SentimentPredictor
import os

st.set_page_config(
    page_title="Sentiment Analysis AI",
    layout="centered"
)

st.title("Sentiment Analysis with DistilBERT")
st.write("Enter a text and the model will predict sentiment.")

# -----------------------------
# Safe model loading
# -----------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join("outputs", "best_model", "model_weights.pt")

    if not os.path.exists(model_path):
        st.error( "Model not found! Please train the model first or add model_weights.pt in outputs/best_model/")
        st.stop()

    return SentimentPredictor()

predictor = load_model()

# -----------------------------
# Input
# -----------------------------
text = st.text_area("Write your sentence here:")

if st.button("Analyze"):
    if text.strip():
        result = predictor.predict_one(text)

        st.subheader("Result")
        st.write(f"**Prediction:** {result['label']}")
        st.write(f"**Confidence:** {result['confidence']:.2f}")

        st.progress(float(result["confidence"]))

        st.write("### Probabilities")
        st.json(result["proba"])
    else:
        st.warning("Please enter text first")

# -----------------------------
# Examples
# -----------------------------
st.markdown("---")
st.write("### Try examples")

examples = [
    "I love this movie, it was amazing!",
    "This was the worst experience ever.",
    "It was okay, nothing special.",
    "Absolutely fantastic performance!",
]

for ex in examples:
    if st.button(ex):
        result = predictor.predict_one(ex)
        st.success(f"{result['label']} ({result['confidence']:.2f})")