"""
model.py
────────
Sentiment Classifier built on top of a pre-trained Transformer Encoder.

Architecture
────────────
  [CLS] token embedding  ← produced by Transformer encoder
        │
   Dropout
        │
   MLP Classifier Head
        │
   Logits → Cross-Entropy Loss
"""

import logging
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

logger = logging.getLogger(__name__)


class SentimentClassifier(nn.Module):
    """
    Transformer Encoder + Classification Head
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        classifier_dropout: float = 0.1,
        freeze_encoder_layers: int = 0,
    ):
        super().__init__()

        self.model_name = model_name
        self.num_labels = num_labels

        logger.info(f"Loading pre-trained encoder: {model_name}")

        config = AutoConfig.from_pretrained(model_name)

        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        hidden_size = config.hidden_size

        # ── Freeze layers if needed ─────────────────────────────
        if freeze_encoder_layers > 0:
            self._freeze_bottom_layers(freeze_encoder_layers)

        # ── Classification head ────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout / 2),
            nn.Linear(hidden_size // 2, num_labels),
        )

        self._init_classifier()

    # ─────────────────────────────────────────────────────────────
    # FORWARD
    # ─────────────────────────────────────────────────────────────
    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        labels=None,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # CLS token
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(cls_embedding)

        out = {"logits": logits}

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            out["loss"] = loss_fn(logits, labels)

        return out

    # ─────────────────────────────────────────────────────────────
    # PREDICTION
    # ─────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def predict_proba(self, input_ids, attention_mask, token_type_ids=None):
        logits = self.forward(input_ids, attention_mask, token_type_ids)["logits"]
        return torch.softmax(logits, dim=-1)

    @torch.inference_mode()
    def predict(self, input_ids, attention_mask, token_type_ids=None):
        return self.predict_proba(input_ids, attention_mask, token_type_ids).argmax(dim=-1)

    # ─────────────────────────────────────────────────────────────
    # FREEZE LOGIC (FIXED FOR BERT + DISTILBERT)
    # ─────────────────────────────────────────────────────────────
    def _freeze_bottom_layers(self, n: int):

        # embeddings
        if hasattr(self.encoder, "embeddings"):
            for p in self.encoder.embeddings.parameters():
                p.requires_grad = False

        # BERT / RoBERTa style
        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            layers = self.encoder.encoder.layer

        # DistilBERT style (IMPORTANT FIX)
        elif hasattr(self.encoder, "transformer"):
            layers = self.encoder.transformer.layer

        else:
            raise ValueError("Unsupported transformer architecture")

        for layer in layers[:n]:
            for p in layer.parameters():
                p.requires_grad = False

        logger.info(f"Frozen {n} encoder layers")

    # ─────────────────────────────────────────────────────────────
    def _init_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ─────────────────────────────────────────────────────────────
    def save(self, path: str):
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), f"{path}/model_weights.pt")
        self.encoder.config.save_pretrained(path)
        logger.info(f"Saved → {path}")

    @classmethod
    def load(cls, path: str, model_name: str, num_labels: int = 2, device: str = "cpu"):
        model = cls(model_name=model_name, num_labels=num_labels)
        state = torch.load(f"{path}/model_weights.pt", map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        logger.info(f"Loaded model from {path}")
        return model

    def num_parameters(self, trainable_only: bool = False):
        params = (
            p for p in self.parameters()
            if (p.requires_grad if trainable_only else True)
        )
        return sum(p.numel() for p in params)