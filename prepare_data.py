"""
prepare_data.py
───────────────
Download and prepare a public sentiment dataset for training.

Supported Datasets
──────────────────
  • IMDB       – 50 000 movie reviews (25K train / 25K test), binary sentiment
  • SST-2      – Stanford Sentiment Treebank, 67 349 sentences, from GLUE
  • Yelp Polarity – 560 000 reviews (large-scale option)

Usage
─────
  python src/prepare_data.py --dataset imdb
  python src/prepare_data.py --dataset sst2
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset       # HuggingFace datasets

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def prepare_imdb(val_frac: float = 0.1, seed: int = 42):
    logger.info("Downloading IMDB dataset …")
    ds = load_dataset("imdb")

    # IMDB already has train / test; we carve a validation set from train
    df_train_full = ds["train"].to_pandas()[["text", "label"]]
    df_test       = ds["test"].to_pandas()[["text", "label"]]

    df_train, df_val = train_test_split(
        df_train_full, test_size=val_frac, stratify=df_train_full["label"], random_state=seed
    )

    _save(df_train, df_val, df_test)
    _print_stats(df_train, df_val, df_test)


def prepare_sst2(seed: int = 42):
    logger.info("Downloading SST-2 dataset …")
    ds = load_dataset("glue", "sst2")

    df_train = ds["train"].to_pandas()[["sentence", "label"]].rename(columns={"sentence": "text"})
    df_val   = ds["validation"].to_pandas()[["sentence", "label"]].rename(columns={"sentence": "text"})

    # SST-2 test labels are hidden; split off a portion of train as held-out test
    df_train, df_test = train_test_split(
        df_train, test_size=0.1, stratify=df_train["label"], random_state=seed
    )

    _save(df_train, df_val, df_test)
    _print_stats(df_train, df_val, df_test)


def prepare_yelp(val_frac: float = 0.05, test_frac: float = 0.05, seed: int = 42):
    logger.info("Downloading Yelp Polarity dataset …")
    ds    = load_dataset("yelp_polarity")
    df    = ds["train"].to_pandas()[["text", "label"]]

    df_train, df_tmp = train_test_split(
        df, test_size=val_frac + test_frac, stratify=df["label"], random_state=seed
    )
    df_val, df_test  = train_test_split(
        df_tmp, test_size=test_frac / (val_frac + test_frac),
        stratify=df_tmp["label"], random_state=seed
    )

    _save(df_train, df_val, df_test)
    _print_stats(df_train, df_val, df_test)


def _save(train, val, test):
    for name, df in [("train", train), ("val", val), ("test", test)]:
        path = DATA_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        logger.info(f"  Saved {len(df):>7,} rows → {path}")


def _print_stats(train, val, test):
    for name, df in [("TRAIN", train), ("VAL", val), ("TEST", test)]:
        pos = (df["label"] == 1).sum()
        neg = (df["label"] == 0).sum()
        logger.info(
            f"{name:5s} | total={len(df):>7,} | "
            f"pos={pos:>6,} ({pos/len(df):.1%}) | "
            f"neg={neg:>6,} ({neg/len(df):.1%})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="imdb",
                        choices=["imdb", "sst2", "yelp"],
                        help="Dataset to prepare")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    if args.dataset == "imdb":
        prepare_imdb(seed=args.seed)
    elif args.dataset == "sst2":
        prepare_sst2(seed=args.seed)
    elif args.dataset == "yelp":
        prepare_yelp(seed=args.seed)

    logger.info("Done! You can now run: python src/train.py config.yaml")
