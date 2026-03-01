"""
Step 1 — Data Cleaning & Preprocessing Pipeline.

Handles:
  - Loading raw Twitter customer support data
  - Pairing inbound (customer) tweets with outbound (company) responses
  - Text cleaning: remove URLs, mentions, emojis, special chars
  - Lemmatization via spaCy
  - Class balancing: keep top-N categories
  - Save cleaned data for downstream modeling

Usage:
    python -m src.preprocessing
"""
import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import (
    RAW_CSV, PROCESSED_CSV, DATA_PROCESSED_DIR,
    SAMPLE_SIZE, TOP_N_CATEGORIES, TEST_SIZE, RANDOM_STATE
)

# ─── Try to load spaCy; fall back to simple splitting ────────────────────────
try:
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    USE_SPACY = True
except Exception:
    USE_SPACY = False
    print("[WARN] spaCy model not found. Using basic tokenization. "
          "Run: python -m spacy download en_core_web_sm")


# ─── Text Cleaning ──────────────────────────────────────────────────────────

def remove_urls(text: str) -> str:
    """Remove HTTP/HTTPS URLs."""
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def remove_mentions(text: str) -> str:
    """Remove @mentions."""
    return re.sub(r"@\w+", "", text)


def remove_emojis(text: str) -> str:
    """Remove emoji characters."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def remove_special_chars(text: str) -> str:
    """Keep only letters, numbers, and basic punctuation."""
    return re.sub(r"[^a-zA-Z0-9\s.,!?']", "", text)


def lemmatize(text: str) -> str:
    """Lemmatize text using spaCy (or return lowercased text as fallback)."""
    if USE_SPACY:
        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc if not token.is_stop and len(token.text) > 1])
    return text.lower()


def clean_text(text: str) -> str:
    """Full cleaning pipeline for a single text string."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = remove_urls(text)
    text = remove_mentions(text)
    text = remove_emojis(text)
    text = remove_special_chars(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = lemmatize(text)
    return text


# ─── Data Loading & Pairing ─────────────────────────────────────────────────

def load_and_pair_data(csv_path: str = RAW_CSV, sample_size: int = SAMPLE_SIZE) -> pd.DataFrame:
    """
    Load the Twitter Customer Support dataset and pair
    inbound customer tweets with outbound company responses.

    Returns a DataFrame with columns:
      - tweet_id, customer_text, company_text, category
    """
    print(f"[INFO] Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Raw dataset shape: {df.shape}")

    # Separate inbound (customer) and outbound (company) tweets
    inbound = df[df["inbound"] == True].copy()
    outbound = df[df["inbound"] == False].copy()

    print(f"[INFO] Inbound tweets: {len(inbound)}, Outbound tweets: {len(outbound)}")

    # Pair: customer tweet -> company response
    # outbound['in_response_to_tweet_id'] tells us which customer tweet it replies to
    outbound = outbound.rename(columns={
        "text": "company_text",
        "author_id": "company_id",
        "tweet_id": "response_tweet_id"
    })

    inbound = inbound.rename(columns={
        "text": "customer_text",
    })

    # Merge: find pairs where company responded to a customer tweet
    paired = inbound.merge(
        outbound[["in_response_to_tweet_id", "company_text", "company_id"]],
        left_on="tweet_id",
        right_on="in_response_to_tweet_id",
        how="inner"
    )

    # The 'company_id' (e.g., AppleSupport, AmazonHelp) is our category label
    paired["category"] = paired["company_id"]

    paired = paired[["tweet_id", "customer_text", "company_text", "category"]].dropna()
    print(f"[INFO] Paired tweets: {len(paired)}")

    # Sample if needed
    if sample_size and len(paired) > sample_size:
        paired = paired.sample(n=sample_size, random_state=RANDOM_STATE)
        print(f"[INFO] Sampled to {sample_size} rows")

    return paired


# ─── Full Preprocessing Pipeline ────────────────────────────────────────────

def preprocess_pipeline(csv_path: str = RAW_CSV) -> pd.DataFrame:
    """
    End-to-end preprocessing:
      1. Load & pair tweets
      2. Keep top-N categories
      3. Clean text
      4. Save processed data
    """
    paired = load_and_pair_data(csv_path)

    # Keep only top-N categories
    top_cats = paired["category"].value_counts().head(TOP_N_CATEGORIES).index.tolist()
    paired = paired[paired["category"].isin(top_cats)].copy()
    print(f"[INFO] Keeping top {TOP_N_CATEGORIES} categories: {top_cats}")
    print(f"[INFO] Rows after filtering: {len(paired)}")

    # Print class distribution
    print("\n[INFO] Class distribution:")
    print(paired["category"].value_counts())

    # Clean text
    print("\n[INFO] Cleaning customer text...")
    paired["cleaned_text"] = paired["customer_text"].apply(clean_text)

    # Also clean company responses (for semantic search later)
    print("[INFO] Cleaning company responses...")
    paired["cleaned_response"] = paired["company_text"].apply(clean_text)

    # Remove empty rows after cleaning
    paired = paired[paired["cleaned_text"].str.len() > 5].copy()

    # Save
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    paired.to_csv(PROCESSED_CSV, index=False)
    print(f"\n[INFO] Saved processed data to {PROCESSED_CSV}")
    print(f"[INFO] Final shape: {paired.shape}")

    return paired


def load_processed_data() -> pd.DataFrame:
    """Load the processed dataset. Run preprocessing if not found."""
    if os.path.exists(PROCESSED_CSV):
        return pd.read_csv(PROCESSED_CSV)
    print("[INFO] Processed data not found. Running preprocessing pipeline...")
    return preprocess_pipeline()


def get_train_test_split(df: pd.DataFrame = None):
    """Return stratified train/test split of cleaned data."""
    if df is None:
        df = load_processed_data()

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(df["category"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_text"].values,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    return X_train, X_test, y_train, y_test, le


if __name__ == "__main__":
    preprocess_pipeline()
