import subprocess
import time

def run_step(command, description):
    print(f"\n{'='*60}")
    print(f"  STEP: {description}")
    print(f"{'='*60}\n")
    start = time.time()
    try:
        subprocess.run(command, check=True, shell=True)
        print(f"\n[SUCCESS] {description} completed in {time.time() - start:.2f}s")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {description} failed with return code {e.returncode}")
        exit(1)

def main():
    print("Starting 10/10 Upgrade Training Sequence...")
    
    # Preprocessing (Already running in background, but this script assumes fresh start or continues)
    # If run manually, uncomment the next line:
    # run_step("python -m src.preprocessing", "Data Preprocessing (200k rows)")
    
    # 1. TF-IDF + Logistic Regression
    run_step("python -m src.model_tfidf_lr", "Training TF-IDF + Logistic Regression")
    
    # 2. Word2Vec + XGBoost
    run_step("python -m src.model_w2v_xgb", "Training Word2Vec + XGBoost")
    
    # 3. DistilBERT Fine-tuning
    run_step("python -m src.model_bert", "Fine-tuning DistilBERT (This may take a while)")
    
    # 4. Semantic Search Index w/ Cross-Encoder
    run_step("python -m src.semantic_search", "Building FAISS Index & Cross-Encoder")
    
    print("\n" + "="*60)
    print("  ALL STEPS COMPLETED SUCCESSFULLY")
    print("="*60)

if __name__ == "__main__":
    main()
