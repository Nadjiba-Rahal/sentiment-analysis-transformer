# 🧠 Sentiment Analysis with BERT Fine-Tuning

A production-grade, end-to-end sentiment analysis pipeline built on top of a
**pre-trained Transformer Encoder (BERT)**, following every principle of modern NLP:
tokenization, contextual embeddings, discriminative fine-tuning, and extrinsic evaluation.

---

## Architecture Overview

```
Raw Text
   │
   ▼
┌─────────────────────────────────┐
│       Text Normalization        │  HTML stripping, URL removal,
│       (preprocess.py)           │  lowercasing, whitespace collapse
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│   Sub-word Tokenization         │  WordPiece (BERT) splits rare words
│   (SentimentTokenizer)          │  into sub-word units → integer IDs
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  BERT Transformer Encoder       │  12 layers × 12 attention heads
│  Contextual Embeddings          │  Each word's vector changes based
│  (model.py)                     │  on ALL surrounding words
└──────────────┬──────────────────┘
               │  [CLS] token embedding (768-dim)
               ▼
┌─────────────────────────────────┐
│  Discriminative Classifier      │  Linear(768→384) → GELU → Linear(384→2)
│  Head                           │  Directly learns the boundary between
└──────────────┬──────────────────┘  Positive and Negative classes
               │
               ▼
          NEGATIVE / POSITIVE
          + confidence score
```

---

## Project Structure

```
sentiment_analysis/
├── config.yaml               ← All hyper-parameters in one place
├── requirements.txt
├── src/
│   ├── prepare_data.py       ← Download & split IMDB / SST-2 / Yelp
│   ├── preprocess.py         ← Normalization + tokenization pipeline
│   ├── dataset.py            ← PyTorch Dataset & DataLoader factory
│   ├── model.py              ← BERT + classifier head (fine-tunable)
│   ├── train.py              ← Discriminative fine-tuning loop
│   ├── metrics.py            ← Accuracy, F1, ROC-AUC, MCC, CM
│   ├── visualize.py          ← Training curves, confusion matrix, PR/ROC
│   └── predict.py            ← Production inference interface
├── data/                     ← train.csv / val.csv / test.csv (auto-generated)
└── outputs/
    ├── best_model/           ← Saved fine-tuned weights
    ├── logs/                 ← training_history.json
    └── evaluation_report.json
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data (IMDB, SST-2, or Yelp)
```bash
# Download and split IMDB (50K reviews, ~2 min)
python src/prepare_data.py --dataset imdb

# Or use the Stanford Sentiment Treebank (SST-2)
python src/prepare_data.py --dataset sst2
```

### 3. Fine-Tune the Model
```bash
python src/train.py config.yaml
```

The trainer will:
- Apply **discriminative learning rates** (lower LR for early layers)
- Use **linear warm-up + decay** scheduling
- Save the **best checkpoint** by validation F1
- Apply **early stopping** to prevent overfitting
- Run **extrinsic evaluation** on the held-out test set

### 4. Predict Sentiment
```bash
# Single text
python src/predict.py "This movie was absolutely phenomenal!"

# Interactive mode
python src/predict.py --interactive

# Batch file (one review per line)
python src/predict.py --file reviews.txt
```

### 5. Visualise Results
```bash
python src/visualize.py
# → outputs/training_curves.png
# → outputs/confusion_matrix.png
# → outputs/pr_roc_curves.png
```

---

## Design Decisions

### Why a Transformer Encoder?
BERT processes every token in relation to every other token simultaneously
(bidirectional attention), producing **contextual embeddings** where the same
word gets a different representation depending on context. This captures nuances
like negation ("not good") that bag-of-words models miss entirely.

### Why Discriminative Fine-Tuning?
Instead of a single global learning rate, we apply **layer-wise decaying LRs**:
- Embedding layer: `LR × 0.1` — preserve general vocabulary knowledge
- Early encoder layers: `LR × 0.1–0.5` — preserve syntactic patterns
- Late encoder layers: `LR × 0.5–1.0` — adapt to sentiment features
- Classifier head: `LR × 1.0` — learn the task boundary freely

This prevents **catastrophic forgetting** while maximising task-specific adaptation.

### Why Extrinsic Evaluation on a Held-Out Test Set?
The test set is never touched during training or hyper-parameter selection.
Only after training is complete do we evaluate on it — giving an unbiased
estimate of real-world performance.

---

## Expected Results (BERT-base on IMDB)

| Metric    | Score  |
|-----------|--------|
| Accuracy  | ~93 %  |
| F1        | ~93 %  |
| ROC-AUC   | ~98 %  |
| MCC       | ~0.86  |

---

## Configuration Reference (`config.yaml`)

| Key | Description |
|-----|-------------|
| `model.backbone` | HuggingFace model name (`bert-base-uncased`, `distilbert-base-uncased`, `roberta-base`) |
| `tokenizer.max_length` | Maximum token sequence length (256–512) |
| `training.learning_rate` | Base LR for the classifier head (typically `1e-5`–`5e-5`) |
| `training.epochs` | Number of fine-tuning epochs (3–5 is usually sufficient) |
| `training.warmup_ratio` | Fraction of steps for LR warm-up (0.06–0.1) |
| `training.fp16` | Enable mixed-precision training (requires CUDA GPU) |

---

## Swapping the Backbone

Change one line in `config.yaml`:
```yaml
model:
  backbone: "distilbert-base-uncased"   # 40 % faster, ~97 % of BERT's accuracy
  # backbone: "roberta-base"            # Often outperforms BERT on sentiment
  # backbone: "bert-large-uncased"      # Larger, slower, higher ceiling
```

No other code changes needed.
