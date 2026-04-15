"""
model.py
────────
Sentiment Classifier built on top of a pre-trained Transformer Encoder.

Architecture
────────────
  [CLS] token embedding  ← produced by BERT's contextual encoder
        │
   Dropout (regularization)
        │
   Linear(hidden_size → num_labels)   ← discriminative classifier head
        │
   Softmax / Cross-Entropy loss

Why a Transformer Encoder?
  • BERT produces *contextual* embeddings: each word's vector changes
    depending on every surrounding word (bidirectional attention).
  • We exploit the rich representations from pre-training on 3.3 B words
    and fine-tune only the top layers for our specific sentiment task.
  • The [CLS] token is trained during fine-tuning to summarise the entire
    input sequence for classification.
"""

import logging
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

logger = logging.getLogger(__name__)


class SentimentClassifier(nn.Module):
    """
    Fine-tunable Transformer Encoder + linear classification head.

    Parameters
    ----------
    model_name : str
        HuggingFace model hub identifier (e.g. 'bert-base-uncased').
    num_labels : int
        Number of output classes (2 for binary sentiment).
    classifier_dropout : float
        Dropout probability on the [CLS] representation.
    freeze_encoder_layers : int
        Number of bottom encoder layers to freeze (0 = fine-tune all).
        Freezing early layers speeds up training when data is scarce.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        classifier_dropout: float = 0.1,
        freeze_encoder_layers: int = 0,
    ):
        super().__init__()
        self.num_labels = num_labels

        # ── Pre-trained Encoder ──────────────────────────────────────────────
        logger.info(f"Loading pre-trained encoder: {model_name}")
        config = AutoConfig.from_pretrained(
            model_name,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        hidden_size  = config.hidden_size          # 768 for bert-base

        # ── Optional Layer Freezing ──────────────────────────────────────────
        if freeze_encoder_layers > 0:
            self._freeze_bottom_layers(freeze_encoder_layers)

        # ── Discriminative Classifier Head ───────────────────────────────────
        # Maps the [CLS] contextual embedding → sentiment logits
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout / 2),
            nn.Linear(hidden_size // 2, num_labels),
        )

        # Initialise the new head with small weights
        self._init_classifier()

    # ── Forward Pass ────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        """
        Args
        ────
        input_ids      : [batch, seq_len]  – token indices
        attention_mask : [batch, seq_len]  – 1 for real tokens, 0 for padding
        token_type_ids : [batch, seq_len]  – segment IDs (0 for single sent.)
        labels         : [batch]           – ground-truth class indices

        Returns
        ───────
        dict with keys:
          'loss'   (if labels provided) – scalar cross-entropy loss
          'logits'                      – [batch, num_labels] raw scores
        """
        # ① Run the Transformer Encoder – produces contextual embeddings
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # ② Extract the [CLS] token representation (index 0)
        #    This vector encodes the full-sequence meaning after fine-tuning
        cls_embedding = outputs.last_hidden_state[:, 0, :]   # [batch, hidden]

        # ③ Pass through the discriminative classifier head → logits
        logits = self.classifier(cls_embedding)               # [batch, num_labels]

        result = {"logits": logits}

        # ④ Compute cross-entropy loss when labels are supplied (training)
        if labels is not None:
            loss_fn      = nn.CrossEntropyLoss()
            result["loss"] = loss_fn(logits, labels)

        return result

    # ── Prediction Helpers ───────────────────────────────────────────────────

    @torch.inference_mode()
    def predict_proba(self, input_ids, attention_mask, token_type_ids=None):
        """Return class probabilities (softmax over logits)."""
        out = self.forward(input_ids, attention_mask, token_type_ids)
        return torch.softmax(out["logits"], dim=-1)

    @torch.inference_mode()
    def predict(self, input_ids, attention_mask, token_type_ids=None):
        """Return predicted class indices."""
        proba = self.predict_proba(input_ids, attention_mask, token_type_ids)
        return proba.argmax(dim=-1)

    # ── Utilities ────────────────────────────────────────────────────────────

    def _freeze_bottom_layers(self, n: int):
        """Freeze embeddings + first n encoder layers."""
        # Embeddings
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
        # Encoder layers
        for layer in self.encoder.encoder.layer[:n]:
            for param in layer.parameters():
                param.requires_grad = False
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        logger.info(
            f"Froze {n} encoder layers. "
            f"Trainable params: {trainable:,} / {total:,}"
        )

    def _init_classifier(self):
        """Xavier-uniform init for the new linear layers."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def save(self, path: str):
        """Save model weights + encoder config to directory."""
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), f"{path}/model_weights.pt")
        self.encoder.config.save_pretrained(path)
        logger.info(f"Model saved → {path}")

    @classmethod
    def load(cls, path: str, model_name: str, num_labels: int = 2, device: str = "cpu"):
        """Load model from a saved directory."""
        model = cls(model_name=model_name, num_labels=num_labels)
        state = torch.load(f"{path}/model_weights.pt", map_location=device)
        model.load_state_dict(state)
        model.eval()
        logger.info(f"Model loaded from {path}")
        return model

    def num_parameters(self, trainable_only: bool = False) -> int:
        params = self.parameters() if not trainable_only else (
            p for p in self.parameters() if p.requires_grad
        )
        return sum(p.numel() for p in params)
