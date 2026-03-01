"""
Unified inference pipeline — brings all components together.

Loads a trained classifier (TF-IDF+LR by default) and the FAISS
semantic search engine. Returns predicted category, confidence,
and a list of suggested solutions from similar resolved tickets.

Usage:
    python -m src.pipeline
"""
import os
import joblib
import numpy as np
from src.config import MODELS_DIR, FAISS_TOP_K
from src.preprocessing import clean_text


class SupportPipeline:
    """
    Customer Support Ticket Auto-Resolver.

    Combines intent classification with semantic search to return:
      - Predicted category
      - Confidence score
      - Similar resolved tickets with suggested solutions
    """

    def __init__(self, classifier_type: str = "tfidf_lr"):
        """
        Args:
            classifier_type: One of "tfidf_lr", "w2v_xgb", "bert", or "ensemble".
        """
        from src.config import logger
        self.logger = logger
        self.classifier_type = classifier_type
        
        # If ensemble, we need to load both TF-IDF and BERT
        if classifier_type == "ensemble":
            self._load_tfidf()
            self._load_bert()
        else:
            self._load_classifier()
            
        self._load_semantic_search()
        self._load_llm()

    def _load_llm(self):
        """Load the Ollama LLM generator."""
        from src.llm import OllamaGenerator
        try:
            self.llm = OllamaGenerator()
            self.logger.info("Ollama LLM generator integrated")
        except Exception as e:
            self.logger.error(f"Failed to load Ollama generator: {e}")
            self.llm = None

    def _load_tfidf(self):
        """Helper to load TF-IDF components."""
        le_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
        self.le = joblib.load(le_path) if os.path.exists(le_path) else None
        
        model_path = os.path.join(MODELS_DIR, "tfidf_lr_model.pkl")
        vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
        if os.path.exists(model_path) and os.path.exists(vec_path):
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vec_path)
            self.logger.info("Loaded TF-IDF + LogReg components")
        else:
            self.logger.warning("TF-IDF components missing")
            self.model = None
            self.vectorizer = None

    def _load_bert(self):
        """Helper to load BERT components."""
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
        import torch
        bert_dir = os.path.join(MODELS_DIR, "bert_model")
        if os.path.exists(bert_dir):
            self.bert_model = DistilBertForSequenceClassification.from_pretrained(bert_dir)
            self.bert_tokenizer = DistilBertTokenizerFast.from_pretrained(bert_dir)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.bert_model.to(self.device)
            self.bert_model.eval()
            self.logger.info(f"Loaded BERT model on {self.device}")
        else:
            self.logger.warning("BERT model missing")
            self.bert_model = None

    def _load_classifier(self):
        """Load the appropriate classifier model."""
        le_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
        self.le = joblib.load(le_path) if os.path.exists(le_path) else None

        if self.classifier_type == "tfidf_lr":
            self._load_tfidf()
        elif self.classifier_type == "w2v_xgb":
            from gensim.models import Word2Vec
            model_path = os.path.join(MODELS_DIR, "xgb_model.pkl")
            w2v_path = os.path.join(MODELS_DIR, "word2vec.model")
            if os.path.exists(model_path) and os.path.exists(w2v_path):
                self.model = joblib.load(model_path)
                self.w2v_model = Word2Vec.load(w2v_path)
                self.logger.info("Loaded Word2Vec + XGBoost model")
            else:
                self.logger.warning("W2V+XGB model missing")
                self.model = None
                self.w2v_model = None
        elif self.classifier_type == "bert":
            self._load_bert()
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")

    def _load_semantic_search(self):
        """Load FAISS semantic search engine."""
        from src.semantic_search import SemanticSearch, FAISS_INDEX_PATH
        if os.path.exists(FAISS_INDEX_PATH):
            self.search_engine = SemanticSearch()
            self.search_engine.load()
        else:
            self.logger.warning("FAISS index not found. Run: python -m src.semantic_search")
            self.search_engine = None

    def classify(self, text: str) -> dict:
        """Classify a ticket into a category with confidence."""
        cleaned = clean_text(text)

        if self.classifier_type == "tfidf_lr":
            return self._classify_tfidf(cleaned)
        elif self.classifier_type == "w2v_xgb":
            return self._classify_w2v(cleaned)
        elif self.classifier_type == "bert":
            return self._classify_bert(cleaned)
        elif self.classifier_type == "ensemble":
            # Weighted Ensemble: 70% BERT + 30% TFIDF
            p_tfidf = self._get_probs_tfidf(cleaned)
            p_bert = self._get_probs_bert(cleaned)
            
            if p_tfidf is None or p_bert is None:
                return {"category": "unknown", "confidence": 0.0}
            
            # Combine probabilities
            ensemble_proba = (0.7 * p_bert) + (0.3 * p_tfidf)
            pred_idx = np.argmax(ensemble_proba)
            
            return {
                "category": self.le.classes_[pred_idx],
                "confidence": float(ensemble_proba[pred_idx]),
                "method": "weighted_ensemble"
            }

    def _get_probs_tfidf(self, cleaned: str):
        if self.model is None or self.vectorizer is None: return None
        X = self.vectorizer.transform([cleaned])
        return self.model.predict_proba(X)[0]

    def _get_probs_bert(self, cleaned: str):
        if self.bert_model is None: return None
        import torch
        from src.config import BERT_MAX_LENGTH
        inputs = self.bert_tokenizer(
            cleaned, return_tensors="pt", truncation=True,
            padding=True, max_length=BERT_MAX_LENGTH
        ).to(self.device)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            return torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

    def _classify_tfidf(self, cleaned: str):
        proba = self._get_probs_tfidf(cleaned)
        if proba is None: return {"category": "unknown", "confidence": 0.0}
        pred_idx = np.argmax(proba)
        return {"category": self.le.classes_[pred_idx], "confidence": float(proba[pred_idx])}

    def _classify_bert(self, cleaned: str):
        proba = self._get_probs_bert(cleaned)
        if proba is None: return {"category": "unknown", "confidence": 0.0}
        pred_idx = np.argmax(proba)
        return {"category": self.le.classes_[pred_idx], "confidence": float(proba[pred_idx])}

    def _classify_w2v(self, cleaned: str):
        if self.model is None: return {"category": "unknown", "confidence": 0.0}
        from src.model_w2v_xgb import text_to_vector
        vec = text_to_vector(cleaned, self.w2v_model).reshape(1, -1)
        proba = self.model.predict_proba(vec)[0]
        pred_idx = np.argmax(proba)
        return {"category": self.le.classes_[pred_idx], "confidence": float(proba[pred_idx])}

    def find_similar(self, text: str, top_k: int = FAISS_TOP_K, rerank: bool = True) -> list:
        """Find similar resolved tickets using re-ranking."""
        if self.search_engine is None:
            return []
        return self.search_engine.search(text, top_k=top_k, rerank=rerank)

    def predict(self, text: str, top_k: int = FAISS_TOP_K, rerank: bool = True) -> dict:
        """
        Full prediction: classify + find similar tickets with re-ranking.
        """
        classification = self.classify(text)
        similar_tickets = self.find_similar(text, top_k=top_k, rerank=rerank)

        suggestions = []
        for ticket in similar_tickets:
            suggestions.append({
                "similar_issue": ticket["customer_text"],
                "suggested_solution": ticket["company_response"],
                "category": ticket["category"],
                "similarity_score": ticket["similarity_score"],
                "rerank_score": ticket.get("rerank_score")
            })

        # 3. Generate customized response with LLM if available, else heuristic
        llm_response = None
        if suggestions:
            if self.llm:
                self.logger.info("Generating customized response with Ollama...")
                llm_response = self.llm.generate_response(text, suggestions)
            else:
                self.logger.info("Ollama unavailable — using heuristic fallback response.")
                llm_response = self._heuristic_response(text, suggestions)

        return {
            "query": text,
            "predicted_category": classification["category"],
            "confidence": round(classification["confidence"], 4),
            "model_used": self.classifier_type,
            "suggested_solutions": suggestions,
            "llm_response": llm_response
        }

    def _heuristic_response(self, query: str, suggestions: list) -> str:
        """
        Synthesize top matches into a rich, query-aware customized response.
        Pulls unique action items from all top suggestions and presents them clearly.
        """
        import re

        top = suggestions[0]
        category = top.get('category', 'your request').replace('_', ' ').title()

        # Collect unique, non-trivial solution sentences across all matches
        seen = set()
        action_points = []
        for s in suggestions:
            sol = s.get('suggested_solution', '').strip()
            # Split on sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', sol)
            for sent in sentences:
                sent = sent.strip()
                key = sent.lower()[:60]
                if len(sent) > 20 and key not in seen:
                    seen.add(key)
                    action_points.append(sent)
                if len(action_points) >= 4:
                    break
            if len(action_points) >= 4:
                break

        steps = "\n".join(f"{i+1}. {pt}" for i, pt in enumerate(action_points))

        return (
            f"Thank you for reaching out. I understand you're having an issue with: \"{query[:120]}\"\n\n"
            f"Based on similar {category} cases we've resolved, here are the recommended steps:\n\n"
            f"{steps}\n\n"
            f"Please try the above steps in order. If the issue persists, reply here and a support "
            f"specialist will review your case directly."
        )


if __name__ == "__main__":
    import json

    print("=" * 60)
    print("  CUSTOMER SUPPORT TICKET AUTO-RESOLVER")
    print("=" * 60)

    pipeline = SupportPipeline(classifier_type="tfidf_lr")

    test_queries = [
        "Where is my order? I've been waiting for 2 weeks!",
        "I can't log into my account, password reset not working",
        "Your app keeps crashing on my iPhone",
        "I was charged twice for the same item",
        "How do I cancel my subscription?",
    ]

    for query in test_queries:
        result = pipeline.predict(query)
        print(f"\n{'─' * 50}")
        print(f"  Query: {result['query']}")
        print(f"  Category: {result['predicted_category']} "
              f"(confidence: {result['confidence']})")
        print(f"  Model: {result['model_used']}")
        if result["suggested_solutions"]:
            print(f"  Top suggestion:")
            top = result["suggested_solutions"][0]
            print(f"    Issue: {top['similar_issue'][:70]}...")
            print(f"    Solution: {top['suggested_solution'][:70]}...")
