"""
Model 1: TF-IDF + Logistic Regression (Baseline).

This is the simplest and fastest model. Often surprisingly competitive
for text classification tasks.

Usage:
    python -m src.model_tfidf_lr
"""
import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from src.config import (
    MODELS_DIR, TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, LR_MAX_ITER, RANDOM_STATE
)
from src.preprocessing import get_train_test_split, load_processed_data


MODEL_PATH = os.path.join(MODELS_DIR, "tfidf_lr_model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")


def train():
    """Train TF-IDF + Logistic Regression pipeline."""
    print("=" * 60)
    print("  MODEL 1: TF-IDF + Logistic Regression")
    print("=" * 60)

    X_train, X_test, y_train, y_test, le = get_train_test_split()

    # TF-IDF Vectorization
    print(f"\n[INFO] Fitting TF-IDF (max_features={TFIDF_MAX_FEATURES}, "
          f"ngram_range={TFIDF_NGRAM_RANGE})...")
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Logistic Regression
    print(f"[INFO] Training Logistic Regression (class_weight='balanced', max_iter={LR_MAX_ITER})...")
    model = LogisticRegression(
        max_iter=LR_MAX_ITER,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        solver="lbfgs",
        multi_class="multinomial"
    )
    model.fit(X_train_tfidf, y_train)

    # Evaluation
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n[RESULT] Accuracy: {acc:.4f}")
    print("\n[RESULT] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))
    print(f"\n[INFO] Model saved to {MODEL_PATH}")
    print(f"[INFO] Vectorizer saved to {VECTORIZER_PATH}")

    return model, vectorizer, le, acc


def predict(text: str, model=None, vectorizer=None, le=None):
    """Predict category for a single text input."""
    if model is None:
        model = joblib.load(MODEL_PATH)
    if vectorizer is None:
        vectorizer = joblib.load(VECTORIZER_PATH)
    if le is None:
        le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

    from src.preprocessing import clean_text
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])

    proba = model.predict_proba(X)[0]
    pred_idx = np.argmax(proba)

    return {
        "category": le.classes_[pred_idx],
        "confidence": float(proba[pred_idx]),
        "all_probabilities": {le.classes_[i]: float(p) for i, p in enumerate(proba)}
    }


if __name__ == "__main__":
    model, vec, le, acc = train()
    print("\n--- Sample Predictions ---")
    samples = [
        "Where is my order? I've been waiting for 2 weeks!",
        "I can't log into my account, password reset not working",
        "Your app keeps crashing on my iPhone",
    ]
    for s in samples:
        result = predict(s, model, vec, le)
        print(f"\n  Input: {s}")
        print(f"  Category: {result['category']} (confidence: {result['confidence']:.3f})")
