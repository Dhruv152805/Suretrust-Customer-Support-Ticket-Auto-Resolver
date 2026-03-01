# 🎫 Customer Support Ticket Auto-Resolver

**NLP + ML + LLM-powered** system that automatically classifies customer support tickets and suggests solutions using semantic search over past resolved tickets.

## 🏗️ Architecture

```
User Complaint
     │
     ▼
┌────────────────┐    ┌──────────────────────────────┐
│  Text Cleaning │    │      Model Comparison         │
│  • Remove URLs │    │  ┌──────────────────────────┐ │
│  • Emojis      │───▶│  │ TF-IDF + Logistic Reg.   │ │
│  • Lemmatize   │    │  │ Word2Vec + XGBoost       │ │
│                │    │  │ DistilBERT Fine-tuned    │ │
└────────────────┘    │  └──────────────────────────┘ │
                      └──────────────┬───────────────┘
                                     │
                      ┌──────────────▼───────────────┐
                      │      Semantic Search          │
                      │  Sentence Transformers + FAISS│
                      │  "Find similar resolved tix"  │
                      └──────────────┬───────────────┘
                                     │
                                     ▼
                      ┌──────────────────────────────┐
                      │         API Response          │
                      │  • Predicted Category         │
                      │  • Confidence Score           │
                      │  • Suggested Solutions        │
                      └──────────────────────────────┘
```

## 📂 Project Structure

```
suretrust2/
├── data/
│   ├── raw/                 # Original dataset (auto-downloaded)
│   └── processed/           # Cleaned, paired data
├── models/                  # Saved model artifacts
├── src/
│   ├── config.py            # All paths & hyperparameters
│   ├── download_data.py     # Kaggle dataset download
│   ├── preprocessing.py     # Text cleaning pipeline
│   ├── model_tfidf_lr.py    # Model 1: TF-IDF + Logistic Regression
│   ├── model_w2v_xgb.py     # Model 2: Word2Vec + XGBoost
│   ├── model_bert.py        # Model 3: DistilBERT fine-tuning
│   ├── semantic_search.py   # FAISS vector search
│   ├── pipeline.py          # Unified inference pipeline
│   └── app.py               # FastAPI deployment
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Download Dataset

```bash
python -m src.download_data
```

### 3. Preprocess Data

```bash
python -m src.preprocessing
```

### 4. Train Models

```bash
# Model 1: TF-IDF + Logistic Regression (fast, ~2 min)
python -m src.model_tfidf_lr

# Model 2: Word2Vec + XGBoost (~5 min)
python -m src.model_w2v_xgb

# Model 3: DistilBERT Fine-tuning (GPU recommended, ~30 min)
python -m src.model_bert
```

### 5. Build Semantic Search Index

```bash
python -m src.semantic_search
```

### 6. Run API

```bash
uvicorn src.app:app --reload
```

### 7. Test

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Where is my order?", "model": "tfidf_lr", "top_k": 3}'
```

Or visit **http://127.0.0.1:8000/docs** for the interactive Swagger UI.

## 🚢 Deployment

### Option 1: Docker (Production Recommended)

Requirements: Docker & Docker Compose

```bash
# Build and start the container
docker-compose up --build -d

# View logs
docker-compose logs -f api
```

### Option 2: Production Script (Windows)

```bash
./run_prod.bat
```

This runs the API using `uvicorn` with 4 worker processes for handling concurrent requests.

### Option 3: Manual Production Run (Linux/Mac)

```bash
export PYTHONPATH=.
uvicorn src.app:app --host 0.0.0.0 --port 8000 --workers 4
```

## 📊 Model Comparison

| Model | Speed | Accuracy | GPU Required |
|-------|-------|----------|-------------|
| TF-IDF + LogReg | ⚡ Fast | Good baseline | ❌ No |
| Word2Vec + XGBoost | 🔶 Medium | Better semantics | ❌ No |
| DistilBERT | 🐢 Slower | Best accuracy | ✅ Recommended |

## 📦 Dataset

**Twitter Customer Support** dataset from Kaggle ([link](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)):
- ~2M+ customer-company tweet pairs
- Companies: AppleSupport, AmazonHelp, Uber_Support, etc.
- Natural language complaint data

## 🔧 Configuration

Edit `src/config.py` to adjust:
- `SAMPLE_SIZE` — rows to use (set `None` for full dataset)
- `TOP_N_CATEGORIES` — number of company categories to keep
- Model hyperparameters (learning rate, epochs, etc.)

## 📄 License

For educational/portfolio purposes.
