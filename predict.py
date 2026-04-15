"""
predict.py
──────────
Production Inference Interface

Loads the fine-tuned model and runs real-time or batch sentiment prediction.

Usage (CLI)
───────────
  python src/predict.py "This product is absolutely amazing!"
  python src/predict.py --file reviews.txt
  python src/predict.py --interactive
"""

import sys
import yaml
import argparse
import logging
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from model     import SentimentClassifier
from preprocess import TextNormalizer, SentimentTokenizer

logger = logging.getLogger(__name__)

LABEL_MAP = {0: "NEGATIVE ❌", 1: "POSITIVE ✅"}


class SentimentPredictor:
    """
    High-level inference wrapper.

    Encapsulates:
      1. Text normalization
      2. Tokenization → contextual embeddings (via BERT encoder)
      3. Discriminative classification → probability scores + label
    """

    def __init__(self, config_path: str = "config.yaml"):
        cfg = yaml.safe_load(Path(config_path).read_text())
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)

        mod_cfg = cfg["model"]
        tok_cfg = cfg["tokenizer"]

        # Load pre-processing components
        self.normalizer = TextNormalizer(lowercase=tok_cfg.get("lowercase", True))
        self.tokenizer  = SentimentTokenizer(
            model_name=mod_cfg["backbone"],
            max_length=tok_cfg["max_length"],
        )

        # Load fine-tuned model
        self.model = SentimentClassifier.load(
            path="Nadjiba04/sentiment-distilbert",
            model_name=mod_cfg["backbone"],
            num_labels=mod_cfg["num_labels"],
            device=device_str,
        ).to(self.device)

        logger.info(f"Predictor ready on {self.device}")

    @torch.inference_mode()
    def predict(self, texts: list[str]) -> list[dict]:
        """
        Run the full inference pipeline on a list of raw texts.

        Returns
        ───────
        List of dicts:
          {
            "text"        : original input text,
            "label"       : "POSITIVE" | "NEGATIVE",
            "confidence"  : float in [0, 1],
            "proba"       : {"NEGATIVE": float, "POSITIVE": float}
          }
        """
        # Step 1 – Normalize
        cleaned = self.normalizer.normalize_batch(texts)

        # Step 2 – Tokenize → vectorize (contextual embeddings produced by encoder)
        encoded = self.tokenizer.encode(cleaned, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        # Step 3 – Forward pass through transformer + classifier head
        proba = self.model.predict_proba(**encoded).cpu().numpy()

        results = []
        for raw_text, p in zip(texts, proba):
            pred_idx = int(p.argmax())
            results.append({
                "text":       raw_text,
                "label":      LABEL_MAP[pred_idx],
                "confidence": round(float(p[pred_idx]), 4),
                "proba": {
                    "NEGATIVE": round(float(p[0]), 4),
                    "POSITIVE": round(float(p[1]), 4),
                },
            })
        return results

    def predict_one(self, text: str) -> dict:
        return self.predict([text])[0]


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_result(r: dict):
    bar_len    = 30
    confidence = r["confidence"]
    filled     = int(bar_len * confidence)
    bar        = "█" * filled + "░" * (bar_len - filled)

    print(f"\n  Text       : {r['text'][:80]}")
    print(f"  Prediction : {r['label']}")
    print(f"  Confidence : [{bar}] {confidence:.1%}")
    print(f"  Scores     : NEG={r['proba']['NEGATIVE']:.4f}  "
          f"POS={r['proba']['POSITIVE']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Sentiment Analysis Inference")
    parser.add_argument("text",         nargs="?", help="Text to classify")
    parser.add_argument("--file",       help="Path to a .txt file (one review per line)")
    parser.add_argument("--interactive",action="store_true", help="Enter interactive mode")
    parser.add_argument("--config",     default="config.yaml")
    args = parser.parse_args()

    predictor = SentimentPredictor(config_path=args.config)

    if args.interactive:
        print("\n🧠  Sentiment Analyser  |  type 'quit' to exit\n")
        while True:
            try:
                text = input("  ➤ Enter text: ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue
            _print_result(predictor.predict_one(text))

    elif args.file:
        lines = Path(args.file).read_text().splitlines()
        lines = [l.strip() for l in lines if l.strip()]
        results = predictor.predict(lines)
        for r in results:
            _print_result(r)

    elif args.text:
        _print_result(predictor.predict_one(args.text))

    else:
        # Demo predictions
        samples = [
            "This film was an absolute masterpiece. Every scene was breathtaking!",
            "Terrible service, the product broke after one day. Never buying again.",
            "The hotel was okay. Nothing special, but nothing terrible either.",
            "I can't believe how good this was – exceeded every expectation.",
            "Waste of money. The description was completely misleading.",
        ]
        print("\n── Demo Predictions ──────────────────────────────────\n")
        for r in predictor.predict(samples):
            _print_result(r)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
