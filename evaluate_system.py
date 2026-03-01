import time
import joblib
import numpy as np
import traceback
import torch
from sklearn.metrics import accuracy_score
from src.config import MODELS_DIR, BERT_MAX_LENGTH, logger
from src.preprocessing import get_train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

def evaluate_models():
    output_file = "evaluation_results.txt"
    with open(output_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  SYSTEM EVALUATION REPORT (10/10 POLISH)\n")
        f.write("=" * 60 + "\n")

        # Load Data
        logger.info("Loading test data...")
        _, X_test_full, _, y_test_full, le = get_train_test_split()
        
        # Subsample for speed
        n_samples = 500
        X_test = X_test_full[:n_samples]
        y_test = y_test_full[:n_samples]
        f.write(f"\n[INFO] Evaluated on {n_samples} samples per model for speed.\n")

        results = []
        
        # Track probabilities for ensemble
        probas_tfidf = None
        probas_bert = None

        # 1. TF-IDF + Logistic Regression
        logger.info("Evaluating TF-IDF + Logistic Regression...")
        try:
            model_tfidf = joblib.load(f"{MODELS_DIR}/tfidf_lr_model.pkl")
            vectorizer = joblib.load(f"{MODELS_DIR}/tfidf_vectorizer.pkl")
            
            start = time.time()
            X_test_tfidf = vectorizer.transform(X_test)
            probas_tfidf = model_tfidf.predict_proba(X_test_tfidf)
            y_pred = np.argmax(probas_tfidf, axis=1)
            duration = time.time() - start
            
            acc = accuracy_score(y_test, y_pred)
            results.append({
                "Model": "TF-IDF + LogReg",
                "Accuracy": acc,
                "Latency (per sample)": (duration / n_samples) * 1000
            })
            print(f"  TF-IDF Accuracy: {acc:.4f}")
        except Exception as e:
            logger.error(f"TF-IDF Eval failed: {traceback.format_exc()}")

        # 2. DistilBERT
        logger.info("Evaluating DistilBERT...")
        try:
            bert_dir = f"{MODELS_DIR}/bert_model"
            tokenizer = DistilBertTokenizerFast.from_pretrained(bert_dir)
            model_bert = DistilBertForSequenceClassification.from_pretrained(bert_dir)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_bert.to(device)
            model_bert.eval()

            start = time.time()
            inputs = tokenizer(X_test.tolist(), return_tensors="pt", padding=True, truncation=True, max_length=BERT_MAX_LENGTH).to(device)
            with torch.no_grad():
                outputs = model_bert(**inputs)
            probas_bert = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            y_pred = np.argmax(probas_bert, axis=1)
            duration = time.time() - start
            
            acc = accuracy_score(y_test, y_pred)
            results.append({
                "Model": "DistilBERT",
                "Accuracy": acc, 
                "Latency (per sample)": (duration / n_samples) * 1000
            })
            print(f"  BERT Accuracy: {acc:.4f}")
        except Exception as e:
            logger.error(f"BERT Eval failed: {traceback.format_exc()}")

        # 3. Weighted Ensemble
        logger.info("Evaluating Weighted Ensemble (70% BERT + 30% TFIDF)...")
        try:
            if probas_tfidf is not None and probas_bert is not None:
                start = time.time()
                ensemble_proba = (0.7 * probas_bert) + (0.3 * probas_tfidf)
                y_pred_ensemble = np.argmax(ensemble_proba, axis=1)
                duration_blend = time.time() - start
                
                # Total latency is roughly BERT + small overhead
                total_latency = results[-1]["Latency (per sample)"] + (duration_blend / n_samples) * 1000
                
                acc = accuracy_score(y_test, y_pred_ensemble)
                results.append({
                    "Model": "Weighted Ensemble",
                    "Accuracy": acc, 
                    "Latency (per sample)": total_latency
                })
                print(f"  Ensemble Accuracy: {acc:.4f}")
        except Exception as e:
            logger.error(f"Ensemble Eval failed: {traceback.format_exc()}")

        f.write("\n" + "-" * 80 + "\n")
        f.write(f"{'Model':<25} | {'Accuracy':<10} | {'Latency (ms/req)':<18}\n")
        f.write("-" * 80 + "\n")
        for r in results:
            f.write(f"{r['Model']:<25} | {r['Accuracy']:.4f}     | {r['Latency (per sample)']:.2f}\n")
        f.write("-" * 80 + "\n")

    print(f"\n[INFO] Results written to {output_file}")

if __name__ == "__main__":
    evaluate_models()
