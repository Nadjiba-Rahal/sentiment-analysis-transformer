"""
preprocess.py
─────────────
Text Preprocessing Pipeline
  1. Normalization  – lowercase, strip HTML/markup, collapse whitespace
  2. Tokenization   – via HuggingFace tokenizer (sub-word BPE / WordPiece)
  3. Vectorization  – tokenizer returns input_ids + attention_mask ready
                      for BERT contextual embeddings
"""

import re
import html
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Union

import pandas as pd
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# ── 1. Text Normalization ────────────────────────────────────────────────────

class TextNormalizer:
    """
    Cleans raw text before tokenization.

    Steps (all configurable):
      • HTML entity decoding  (e.g. &amp; → &)
      • Markup tag removal    (<br />, <p>, …)
      • URL/email stripping
      • Lowercase conversion
      • Non-ASCII / emoji control
      • Whitespace collapsing
    """

    _URL_RE    = re.compile(r"https?://\S+|www\.\S+")
    _EMAIL_RE  = re.compile(r"\S+@\S+\.\S+")
    _HTML_RE   = re.compile(r"<[^>]+>")
    _MULTI_WS  = re.compile(r"\s+")

    def __init__(
        self,
        lowercase: bool = True,
        strip_html: bool = True,
        strip_urls: bool = True,
        strip_emails: bool = True,
    ):
        self.lowercase    = lowercase
        self.strip_html   = strip_html
        self.strip_urls   = strip_urls
        self.strip_emails = strip_emails

    def normalize(self, text: str) -> str:
        """Apply full normalization pipeline to a single string."""
        # Decode HTML entities first (&amp; → &)
        text = html.unescape(text)

        if self.strip_html:
            text = self._HTML_RE.sub(" ", text)

        if self.strip_urls:
            text = self._URL_RE.sub(" ", text)

        if self.strip_emails:
            text = self._EMAIL_RE.sub(" ", text)

        if self.lowercase:
            text = text.lower()

        # Collapse multiple whitespace / newlines into a single space
        text = self._MULTI_WS.sub(" ", text).strip()

        return text

    def normalize_batch(self, texts: List[str]) -> List[str]:
        return [self.normalize(t) for t in texts]


# ── 2 & 3. Tokenization + Vectorization (Contextual Embeddings) ─────────────

class SentimentTokenizer:
    """
    Wraps a HuggingFace AutoTokenizer to produce BERT-ready tensors.

    The tokenizer performs:
      • Sub-word segmentation (WordPiece for BERT)
      • Special token injection ([CLS], [SEP])
      • Padding & truncation to a fixed sequence length
      • Attention mask generation (1 = real token, 0 = padding)

    Contextual embeddings are produced by the model itself (BERT encoder);
    the tokenizer only converts text → integer IDs.
    """

    def __init__(self, model_name: str, max_length: int = 256):
        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer  = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def encode(
        self,
        texts: Union[str, List[str]],
        return_tensors: str = "pt",
    ) -> Dict:
        """
        Tokenize + vectorize one or a batch of texts.

        Returns a dict with:
          input_ids      – token IDs  [batch, seq_len]
          attention_mask – 1/0 mask   [batch, seq_len]
          token_type_ids – segment IDs (always 0 for single-sentence tasks)
        """
        if isinstance(texts, str):
            texts = [texts]

        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=return_tensors,
        )
        return encoded

    def decode(self, input_ids) -> List[str]:
        """Convert token IDs back to human-readable strings (for debugging)."""
        return self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size


# ── 4. Data Loading Utility ──────────────────────────────────────────────────

def load_dataset_csv(path: str, text_col: str, label_col: str) -> pd.DataFrame:
    """
    Load a CSV dataset.

    Expected columns: <text_col>, <label_col>
    Label values  : 0 = Negative, 1 = Positive
    """
    df = pd.read_csv(path)

    missing = {text_col, label_col} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    df = df[[text_col, label_col]].dropna()
    df[label_col] = df[label_col].astype(int)

    logger.info(
        f"Loaded {len(df)} samples | "
        f"Pos: {(df[label_col]==1).sum()} | "
        f"Neg: {(df[label_col]==0).sum()}"
    )
    return df


def build_preprocessing_pipeline(config_path: str = "config.yaml"):
    """Factory: returns (normalizer, tokenizer) from a YAML config."""
    cfg = yaml.safe_load(Path(config_path).read_text())
    tok_cfg = cfg["tokenizer"]
    mod_cfg = cfg["model"]

    normalizer = TextNormalizer(lowercase=tok_cfg.get("lowercase", True))
    tokenizer  = SentimentTokenizer(
        model_name=mod_cfg["backbone"],
        max_length=tok_cfg["max_length"],
    )
    return normalizer, tokenizer


# ── Quick sanity-check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sample_texts = [
        "<p>This movie was <b>absolutely FANTASTIC</b>! Loved every minute.</p>",
        "I hated this product. Total waste of money. Visit http://refund.com",
        "It's   an  okay   film,   nothing   special.",
    ]

    norm = TextNormalizer()
    cleaned = norm.normalize_batch(sample_texts)

    print("─── Normalized Texts ───")
    for orig, clean in zip(sample_texts, cleaned):
        print(f"  IN : {orig}")
        print(f"  OUT: {clean}\n")
