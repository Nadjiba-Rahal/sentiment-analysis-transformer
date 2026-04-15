"""
dataset.py
──────────
PyTorch Dataset & DataLoader factory for the sentiment pipeline.

Integrates the TextNormalizer + SentimentTokenizer so that each
__getitem__ call returns fully preprocessed, BERT-ready tensors.
"""

import torch
import logging
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from preprocess import TextNormalizer, SentimentTokenizer, load_dataset_csv

logger = logging.getLogger(__name__)


class SentimentDataset(Dataset):
    """
    Map-style PyTorch Dataset.

    Each item returns:
        input_ids      : LongTensor [seq_len]
        attention_mask : LongTensor [seq_len]
        token_type_ids : LongTensor [seq_len]
        label          : LongTensor []
    """

    def __init__(
        self,
        texts: list,
        labels: list,
        normalizer: TextNormalizer,
        tokenizer: SentimentTokenizer,
    ):
        self.labels     = labels
        self.normalizer = normalizer
        self.tokenizer  = tokenizer

        logger.info(f"Normalizing {len(texts)} texts …")
        self.texts = normalizer.normalize_batch(texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        encoded = self.tokenizer.encode(self.texts[idx], return_tensors="pt")

        return {
            "input_ids":      encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "token_type_ids": encoded.get(
                "token_type_ids",
                torch.zeros_like(encoded["input_ids"])
            ).squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def build_dataloaders(
    cfg: dict,
    normalizer: TextNormalizer,
    tokenizer: SentimentTokenizer,
):
    """
    Load CSVs, apply train/val/test splits, return DataLoader trio.

    If only a single CSV is provided it is split automatically using
    val_split and test_split from the config.
    """
    data_cfg  = cfg["data"]
    train_cfg = cfg["training"]

    # ── Load ────────────────────────────────────────────────────────────────
    df_train = load_dataset_csv(
        data_cfg["train_path"],
        data_cfg["text_col"],
        data_cfg["label_col"],
    )

    df_val = load_dataset_csv(
        data_cfg["val_path"],
        data_cfg["text_col"],
        data_cfg["label_col"],
    )

    df_test = load_dataset_csv(
        data_cfg["test_path"],
        data_cfg["text_col"],
        data_cfg["label_col"],
    )

    def _make_dataset(df: pd.DataFrame) -> SentimentDataset:
        return SentimentDataset(
            texts=df[data_cfg["text_col"]].tolist(),
            labels=df[data_cfg["label_col"]].tolist(),
            normalizer=normalizer,
            tokenizer=tokenizer,
        )

    train_ds = _make_dataset(df_train)
    val_ds   = _make_dataset(df_val)
    test_ds  = _make_dataset(df_test)

    batch_size = train_cfg["batch_size"]

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader  = DataLoader(
        test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    logger.info(
        f"Splits → Train: {len(train_ds)} | "
        f"Val: {len(val_ds)} | Test: {len(test_ds)}"
    )
    return train_loader, val_loader, test_loader


def auto_split_and_build(
    csv_path: str,
    text_col: str,
    label_col: str,
    normalizer: TextNormalizer,
    tokenizer: SentimentTokenizer,
    val_size: float = 0.1,
    test_size: float = 0.1,
    batch_size: int = 32,
    seed: int = 42,
):
    """
    Convenience helper: load one CSV and split it automatically.
    Returns (train_loader, val_loader, test_loader).
    """
    df = load_dataset_csv(csv_path, text_col, label_col)
    texts  = df[text_col].tolist()
    labels = df[label_col].tolist()

    # Stratified split to preserve class balance
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        texts, labels,
        test_size=val_size + test_size,
        stratify=labels,
        random_state=seed,
    )
    relative_test = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=relative_test,
        stratify=y_tmp,
        random_state=seed,
    )

    def _dl(texts, labels, shuffle):
        ds = SentimentDataset(texts, labels, normalizer, tokenizer)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    return _dl(X_train, y_train, True), _dl(X_val, y_val, False), _dl(X_test, y_test, False)
