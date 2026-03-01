"""
Central configuration for the Customer Support Ticket Auto-Resolver.
All paths, hyperparameters, and constants in one place.
"""
import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

RAW_CSV = os.path.join(DATA_RAW_DIR, "twcs.csv")
PROCESSED_CSV = os.path.join(DATA_PROCESSED_DIR, "cleaned_tickets.csv")

# ─── Data Parameters ────────────────────────────────────────────────────────
# Number of rows to sample from the raw dataset (None = use all)
SAMPLE_SIZE = 200_000

# Keep only the top-N most frequent company categories
TOP_N_CATEGORIES = 25  # Increased from 15 for better coverage

# Train/test split ratio
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ─── TF-IDF + Logistic Regression ───────────────────────────────────────────
TFIDF_MAX_FEATURES = 10_000
TFIDF_NGRAM_RANGE = (1, 2)
LR_MAX_ITER = 1000

# ─── Word2Vec + XGBoost ─────────────────────────────────────────────────────
W2V_VECTOR_SIZE = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 2
W2V_EPOCHS = 10
XGB_N_ESTIMATORS = 200
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.1

# ─── BERT (DistilBERT) ──────────────────────────────────────────────────────
BERT_MODEL_NAME = "distilbert-base-uncased"
BERT_MAX_LENGTH = 128
BERT_BATCH_SIZE = 16
BERT_EPOCHS = 3
BERT_LEARNING_RATE = 2e-5

# ─── Semantic Search ────────────────────────────────────────────────────────
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
FAISS_TOP_K = 10  # Retrieve more for the cross-encoder to re-rank
FINAL_TOP_K = 3   # Final solutions to show after re-ranking

# ─── Ollama LLM ─────────────────────────────────────────────────────────────

OLLAMA_MODEL = "llama3.2:1b"
OLLAMA_HOST = "http://localhost:11434"

# ─── Create directories ─────────────────────────────────────────────────────
for d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Logging ────────────────────────────────────────────────────────────────
import logging

def setup_logging(level=logging.INFO):
    """Simple global logging setup."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("suretrust")

logger = setup_logging()
