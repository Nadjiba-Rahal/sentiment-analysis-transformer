import streamlit as st
import pandas as pd
from predict import SentimentPredictor

st.set_page_config(
    page_title="Sentiment Lens | Transformer sentiment analysis",
    page_icon="SL",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { max-width: 1280px; padding-top: 1.5rem; padding-bottom: 1rem; }
    .hero { padding: .5rem 0 1.25rem; border-bottom: 1px solid #d9e2e8; margin-bottom: 1.25rem; }
    .eyebrow { color: #007c83; font-size: .78rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .hero h1 { color: #102a43; font-size: clamp(2rem, 4vw, 3.5rem); line-height: 1; margin: .35rem 0 .7rem; }
    .hero p { color: #486581; font-size: 1rem; max-width: 650px; margin-bottom: 0; }
    .result-card { border-left: 5px solid #007c83; background: #f0f7f7; padding: 1.25rem 1.5rem; margin: 1rem 0; }
    div[data-testid="stMetricValue"] { color: #102a43; }
</style>
<div class="hero">
  <div class="eyebrow">Transformer inference studio</div>
  <h1>Read the feeling<br>behind the words.</h1>
  <p>Explore sentiment predictions from a fine-tuned DistilBERT classifier, with transparent confidence scores for every result.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Safe model loading
# -----------------------------
@st.cache_resource
def load_model():
    try:
        return SentimentPredictor()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

predictor = load_model()

with st.sidebar:
    st.markdown("### Sentiment Lens")
    st.caption("A focused demonstration of production-minded NLP inference.")
    st.divider()
    st.markdown("**Model**  \\n+DistilBERT fine-tuned for binary sentiment")
    st.markdown("**Runtime**  \\n+" + str(predictor.device).upper())
    st.markdown("**Labels**  \\n+Negative / Positive")
    st.divider()
    st.caption("Inputs are normalized before tokenization. URLs, HTML, and excess whitespace are removed.")

single_column, batch_workspace = st.columns([1, 1], gap="large")

with single_column:
    st.subheader("Analyze a text")
    example_options = {
        "Write your own review": "",
        "A great experience": "The product exceeded my expectations and I would recommend it.",
        "A disappointing experience": "The service was slow, frustrating, and not worth the price.",
        "A mixed reaction": "The design is beautiful, but the setup took longer than expected.",
    }
    selected_example = st.selectbox("Start with an example", list(example_options))
    text = st.text_area(
        "Text to classify",
        value=example_options[selected_example],
        height=150,
        placeholder="Share a review, comment, or message...",
        max_chars=2000,
    )

    if st.button("Analyze sentiment", type="primary", use_container_width=True):
        if text.strip():
            result = predictor.predict_one(text)

            sentiment = "Positive" if result["label"].startswith("POSITIVE") else "Negative"
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.metric("Prediction", sentiment, f"{result['confidence']:.1%} confidence")
            st.progress(float(result["confidence"]))
            st.markdown('</div>', unsafe_allow_html=True)

            probability_df = pd.DataFrame({"Sentiment": ["Negative", "Positive"], "Probability": [result["proba"]["NEGATIVE"], result["proba"]["POSITIVE"]]})
            st.bar_chart(probability_df.set_index("Sentiment"), color="#007c83")
        else:
            st.warning("Enter some text before running an analysis.")

with batch_workspace:
    st.subheader("Score a collection of texts")
    st.caption("Upload a CSV containing a column of reviews or comments. Predictions are added without changing your original data.")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        text_columns = batch_df.select_dtypes(include="object").columns.tolist()
        if not text_columns:
            st.error("No text column was found in this CSV.")
        else:
            selected_text_column = st.selectbox("Text column", text_columns)
            if st.button("Analyze CSV", type="primary"):
                texts = batch_df[selected_text_column].fillna("").astype(str).tolist()
                results = predictor.predict(texts)
                result_df = batch_df.copy()
                result_df["sentiment"] = [r["label"].split()[0].lower() for r in results]
                result_df["confidence"] = [r["confidence"] for r in results]
                positive_count = int((result_df["sentiment"] == "positive").sum())
                negative_count = len(result_df) - positive_count
                summary_columns = st.columns(3)
                summary_columns[0].metric("Rows analyzed", len(result_df))
                summary_columns[1].metric("Positive", positive_count)
                summary_columns[2].metric("Negative", negative_count)
                st.dataframe(result_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download predictions",
                    result_df.to_csv(index=False).encode("utf-8"),
                    "sentiment_predictions.csv",
                    "text/csv",
                )

# -----------------------------
# Examples
# -----------------------------
st.divider()
st.caption("Confidence reflects the model's probability estimate, not a guarantee of correctness.")