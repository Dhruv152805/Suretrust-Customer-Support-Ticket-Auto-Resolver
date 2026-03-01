"""
Model 3: DistilBERT Fine-Tuning for Ticket Classification.

Uses HuggingFace Transformers to fine-tune distilbert-base-uncased
for multi-class intent classification.

Usage:
    python -m src.model_bert
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, classification_report
import joblib
from src.config import (
    MODELS_DIR, BERT_MODEL_NAME, BERT_MAX_LENGTH,
    BERT_BATCH_SIZE, BERT_EPOCHS, BERT_LEARNING_RATE, RANDOM_STATE
)
from src.preprocessing import get_train_test_split

BERT_OUTPUT_DIR = os.path.join(MODELS_DIR, "bert_model")


# ─── Custom Dataset ─────────────────────────────────────────────────────────

class TicketDataset(Dataset):
    """PyTorch Dataset for tokenized ticket data."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    """Compute accuracy for HuggingFace Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


# ─── Training ───────────────────────────────────────────────────────────────

def train():
    """Fine-tune DistilBERT on ticket classification."""
    print("=" * 60)
    print("  MODEL 3: DistilBERT Fine-Tuning")
    print("=" * 60)

    X_train, X_test, y_train, y_test, le = get_train_test_split()
    num_labels = len(le.classes_)

    print(f"[INFO] Number of classes: {num_labels}")
    print(f"[INFO] Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Tokenize
    print(f"\n[INFO] Loading tokenizer: {BERT_MODEL_NAME}...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_MODEL_NAME)

    print("[INFO] Tokenizing texts...")
    train_encodings = tokenizer(
        list(X_train), truncation=True, padding=True, max_length=BERT_MAX_LENGTH
    )
    test_encodings = tokenizer(
        list(X_test), truncation=True, padding=True, max_length=BERT_MAX_LENGTH
    )

    train_dataset = TicketDataset(train_encodings, y_train)
    test_dataset = TicketDataset(test_encodings, y_test)

    # Model
    print(f"[INFO] Loading model: {BERT_MODEL_NAME} with {num_labels} labels...")
    model = DistilBertForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME, num_labels=num_labels
    )

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=BERT_OUTPUT_DIR,
        num_train_epochs=BERT_EPOCHS,
        per_device_train_batch_size=BERT_BATCH_SIZE,
        per_device_eval_batch_size=BERT_BATCH_SIZE * 2,
        learning_rate=BERT_LEARNING_RATE,
        warmup_steps=500,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir=os.path.join(BERT_OUTPUT_DIR, "logs"),
        logging_steps=100,
        seed=RANDOM_STATE,
        report_to="none",  # Disable wandb etc.
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\n[INFO] Starting training...")
    trainer.train()

    # Evaluate
    print("\n[INFO] Evaluating on test set...")
    results = trainer.evaluate()
    print(f"[RESULT] Test Accuracy: {results['eval_accuracy']:.4f}")

    # Detailed classification report
    preds = trainer.predict(test_dataset)
    y_pred = np.argmax(preds.predictions, axis=-1)
    print("\n[RESULT] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save model and tokenizer
    trainer.save_model(BERT_OUTPUT_DIR)
    tokenizer.save_pretrained(BERT_OUTPUT_DIR)
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.pkl"))
    print(f"\n[INFO] Model saved to {BERT_OUTPUT_DIR}")

    return model, tokenizer, le, results["eval_accuracy"]


# ─── Inference ───────────────────────────────────────────────────────────────

def predict(text: str, model=None, tokenizer=None, le=None):
    """Predict category for a single text with confidence score."""
    if model is None:
        model = DistilBertForSequenceClassification.from_pretrained(BERT_OUTPUT_DIR)
    if tokenizer is None:
        tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_OUTPUT_DIR)
    if le is None:
        le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

    from src.preprocessing import clean_text
    cleaned = clean_text(text)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    inputs = tokenizer(
        cleaned, return_tensors="pt", truncation=True,
        padding=True, max_length=BERT_MAX_LENGTH
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        proba = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

    pred_idx = np.argmax(proba)
    return {
        "category": le.classes_[pred_idx],
        "confidence": float(proba[pred_idx]),
        "all_probabilities": {le.classes_[i]: float(p) for i, p in enumerate(proba)}
    }


if __name__ == "__main__":
    model, tokenizer, le, acc = train()
    print("\n--- Sample Predictions ---")
    samples = [
        "Where is my order? I've been waiting for 2 weeks!",
        "I can't log into my account, password reset not working",
        "Your app keeps crashing on my iPhone",
    ]
    for s in samples:
        result = predict(s, model, tokenizer, le)
        print(f"\n  Input: {s}")
        print(f"  Category: {result['category']} (confidence: {result['confidence']:.3f})")
