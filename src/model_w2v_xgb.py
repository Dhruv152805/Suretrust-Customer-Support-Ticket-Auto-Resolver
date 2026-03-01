"""
Model 2: Word2Vec + XGBoost.

Captures semantic relationships between words using Word2Vec embeddings,
then uses XGBoost (gradient boosting) for classification.

Usage:
    python -m src.model_w2v_xgb
"""
import os
import joblib
import numpy as np
from gensim.models import Word2Vec
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from src.config import (
    MODELS_DIR, W2V_VECTOR_SIZE, W2V_WINDOW, W2V_MIN_COUNT, W2V_EPOCHS,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE, RANDOM_STATE
)
from src.preprocessing import get_train_test_split

W2V_MODEL_PATH = os.path.join(MODELS_DIR, "word2vec.model")
XGB_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model.pkl")


def tokenize_texts(texts):
    """Simple whitespace tokenizer for Word2Vec."""
    return [text.split() for text in texts]


def train_word2vec(tokenized_texts):
    """Train a Word2Vec model on the corpus."""
    print(f"[INFO] Training Word2Vec (dim={W2V_VECTOR_SIZE}, window={W2V_WINDOW})...")
    model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=W2V_VECTOR_SIZE,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        workers=4,
        epochs=W2V_EPOCHS,
        seed=RANDOM_STATE
    )
    model.save(W2V_MODEL_PATH)
    print(f"[INFO] Word2Vec model saved. Vocabulary size: {len(model.wv)}")
    return model


def text_to_vector(text: str, w2v_model: Word2Vec) -> np.ndarray:
    """
    Convert a text to its document vector by averaging
    the Word2Vec embeddings of its words.
    """
    words = text.split()
    word_vectors = []
    for word in words:
        if word in w2v_model.wv:
            word_vectors.append(w2v_model.wv[word])
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    return np.zeros(W2V_VECTOR_SIZE)


def texts_to_vectors(texts, w2v_model: Word2Vec) -> np.ndarray:
    """Convert multiple texts to their document vectors."""
    return np.array([text_to_vector(t, w2v_model) for t in texts])


def train():
    """Train Word2Vec + XGBoost pipeline."""
    print("=" * 60)
    print("  MODEL 2: Word2Vec + XGBoost")
    print("=" * 60)

    X_train, X_test, y_train, y_test, le = get_train_test_split()

    # Train Word2Vec on training data
    all_texts = np.concatenate([X_train, X_test])
    tokenized = tokenize_texts(all_texts)
    w2v_model = train_word2vec(tokenized)

    # Convert texts to vectors
    print("[INFO] Converting texts to Word2Vec document vectors...")
    X_train_vec = texts_to_vectors(X_train, w2v_model)
    X_test_vec = texts_to_vectors(X_test, w2v_model)

    print(f"[INFO] Feature matrix shape: {X_train_vec.shape}")

    # Train XGBoost
    print(f"\n[INFO] Training XGBoost (n_estimators={XGB_N_ESTIMATORS}, "
          f"max_depth={XGB_MAX_DEPTH}, lr={XGB_LEARNING_RATE})...")

    num_classes = len(le.classes_)
    xgb = XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        objective="multi:softprob",
        num_class=num_classes,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=1
    )
    xgb.fit(
        X_train_vec, y_train,
        eval_set=[(X_test_vec, y_test)],
        verbose=True
    )

    # Evaluation
    y_pred = xgb.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n[RESULT] Accuracy: {acc:.4f}")
    print("\n[RESULT] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save
    joblib.dump(xgb, XGB_MODEL_PATH)
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))
    print(f"\n[INFO] XGBoost model saved to {XGB_MODEL_PATH}")

    return xgb, w2v_model, le, acc


def predict(text: str, xgb_model=None, w2v_model=None, le=None):
    """Predict category for a single text input."""
    if xgb_model is None:
        xgb_model = joblib.load(XGB_MODEL_PATH)
    if w2v_model is None:
        w2v_model = Word2Vec.load(W2V_MODEL_PATH)
    if le is None:
        le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

    from src.preprocessing import clean_text
    cleaned = clean_text(text)
    vec = text_to_vector(cleaned, w2v_model).reshape(1, -1)

    proba = xgb_model.predict_proba(vec)[0]
    pred_idx = np.argmax(proba)

    return {
        "category": le.classes_[pred_idx],
        "confidence": float(proba[pred_idx]),
        "all_probabilities": {le.classes_[i]: float(p) for i, p in enumerate(proba)}
    }


if __name__ == "__main__":
    xgb, w2v, le, acc = train()
    print("\n--- Sample Predictions ---")
    samples = [
        "Where is my order? I've been waiting for 2 weeks!",
        "I can't log into my account, password reset not working",
        "Your app keeps crashing on my iPhone",
    ]
    for s in samples:
        result = predict(s, xgb, w2v, le)
        print(f"\n  Input: {s}")
        print(f"  Category: {result['category']} (confidence: {result['confidence']:.3f})")
