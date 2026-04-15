"""
predict.py
──────────
Production Inference Interface
"""

import sys
import yaml
import logging
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))

from model import SentimentClassifier
from preprocess import TextNormalizer, SentimentTokenizer

logger = logging.getLogger(__name__)

LABEL_MAP = {0: "NEGATIVE ❌", 1: "POSITIVE ✅"}


class SentimentPredictor:
    def __init__(self, config_path: str = "config.yaml"):

        # ─────────────────────────────────────────────
        # SAFE CONFIG LOADING (FIXED)
        # ─────────────────────────────────────────────
        if Path(config_path).exists():
            cfg = yaml.safe_load(Path(config_path).read_text())
        else:
            logger.warning("config.yaml not found → using default config")

            cfg = {
                "model": {
                    "backbone": "distilbert-base-uncased",
                    "num_labels": 2,
                },
                "tokenizer": {
                    "max_length": 256,
                    "lowercase": True,
                }
            }

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)

        mod_cfg = cfg["model"]
        tok_cfg = cfg["tokenizer"]

        # ─────────────────────────────────────────────
        # PREPROCESS
        # ─────────────────────────────────────────────
        self.normalizer = TextNormalizer(lowercase=tok_cfg.get("lowercase", True))

        self.tokenizer = SentimentTokenizer(
            model_name=mod_cfg["backbone"],
            max_length=tok_cfg["max_length"],
        )

        # ─────────────────────────────────────────────
        # DOWNLOAD MODEL FROM HUGGING FACE (FIXED)
        # ─────────────────────────────────────────────
        model_file = hf_hub_download(
            repo_id="Nadjiba04/sentiment-distilbert",
            filename="model_weights.pt"
        )

        # ─────────────────────────────────────────────
        # LOAD MODEL
        # ─────────────────────────────────────────────
        self.model = SentimentClassifier(
            model_name=mod_cfg["backbone"],
            num_labels=mod_cfg["num_labels"],
        )

        state_dict = torch.load(model_file, map_location=device_str)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Predictor ready on {self.device}")

    # ─────────────────────────────────────────────
    @torch.inference_mode()
    def predict(self, texts: list[str]):

        cleaned = self.normalizer.normalize_batch(texts)

        encoded = self.tokenizer.encode(cleaned, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        proba = self.model.predict_proba(**encoded).cpu().numpy()

        results = []
        for raw_text, p in zip(texts, proba):
            pred_idx = int(p.argmax())

            results.append({
                "text": raw_text,
                "label": LABEL_MAP[pred_idx],
                "confidence": round(float(p[pred_idx]), 4),
                "proba": {
                    "NEGATIVE": round(float(p[0]), 4),
                    "POSITIVE": round(float(p[1]), 4),
                },
            })

        return results

    def predict_one(self, text: str):
        return self.predict([text])[0]