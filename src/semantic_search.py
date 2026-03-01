"""
Step 3 — Semantic Search using Sentence Transformers + FAISS.

Encodes resolved tickets into dense vectors and builds a FAISS index
for fast similarity retrieval.

Usage:
    python -m src.semantic_search
"""
import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.config import (
    MODELS_DIR, SENTENCE_TRANSFORMER_MODEL, CROSS_ENCODER_MODEL,
    FAISS_TOP_K, FINAL_TOP_K
)
from src.preprocessing import load_processed_data

FAISS_INDEX_PATH = os.path.join(MODELS_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(MODELS_DIR, "ticket_metadata.json")


class SemanticSearch:
    """
    Semantic search engine for customer support tickets.

    Encodes resolved tickets using Sentence Transformers and indexes
    them with FAISS for fast nearest-neighbor retrieval.
    """

    def __init__(self, model_name: str = SENTENCE_TRANSFORMER_MODEL, 
                 cross_model_name: str = CROSS_ENCODER_MODEL):
        print(f"[INFO] Loading Sentence Transformer: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        print(f"[INFO] Loading Cross-Encoder: {cross_model_name}...")
        self.cross_encoder = CrossEncoder(cross_model_name)
        
        self.index = None
        self.metadata = []

    def build_index(self, df=None):
        """
        Build FAISS index from resolved tickets.

        Each entry stores:
          - The customer's original complaint
          - The company's response (suggested solution)
          - The category
        """
        if df is None:
            df = load_processed_data()

        print(f"[INFO] Building FAISS index from {len(df)} resolved tickets...")

        # Use customer text for semantic matching
        texts = df["customer_text"].fillna("").tolist()

        # Encode all texts
        print("[INFO] Encoding texts (this may take a few minutes)...")
        embeddings = self.model.encode(
            texts, show_progress_bar=True, batch_size=256
        )
        embeddings = np.array(embeddings, dtype="float32")

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Build index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity after normalization
        self.index.add(embeddings)

        # Store metadata for retrieval
        self.metadata = []
        for _, row in df.iterrows():
            self.metadata.append({
                "customer_text": str(row.get("customer_text", "")),
                "company_response": str(row.get("company_text", "")),
                "category": str(row.get("category", "")),
            })

        print(f"[INFO] FAISS index built with {self.index.ntotal} vectors "
              f"(dim={dimension})")

        # Save
        self.save()
        return self

    def save(self):
        """Save FAISS index and metadata to disk."""
        faiss.write_index(self.index, FAISS_INDEX_PATH)
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)
        print(f"[INFO] Index saved to {FAISS_INDEX_PATH}")
        print(f"[INFO] Metadata saved to {METADATA_PATH}")

    def load(self):
        """Load FAISS index and metadata from disk."""
        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError(
                f"FAISS index not found at {FAISS_INDEX_PATH}. "
                "Run `python -m src.semantic_search` to build it."
            )
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        print(f"[INFO] Loaded FAISS index ({self.index.ntotal} vectors)")
        return self

    def search(self, query: str, top_k: int = FAISS_TOP_K, rerank: bool = True) -> list:
        """
        Find the most similar resolved tickets to a query.
        If rerank=True, use the Cross-Encoder to refine the order.
        """
        if self.index is None:
            self.load()

        # 1. Fast Retrieval with Bi-Encoder (FAISS)
        query_vec = self.model.encode([query])
        query_vec = np.array(query_vec, dtype="float32")
        faiss.normalize_L2(query_vec)

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                entry = self.metadata[idx].copy()
                entry["similarity_score"] = float(score)
                results.append(entry)

        # 2. Precision Re-ranking with Cross-Encoder
        if rerank and results:
            print(f"[INFO] Re-ranking top {len(results)} results with Cross-Encoder...")
            # Pair query with each candidate for re-ranking
            pairs = [[query, r["customer_text"]] for r in results]
            rerank_scores = self.cross_encoder.predict(pairs)
            
            # Update scores and sort
            for i in range(len(results)):
                # Cross-encoder scores are usually logits (can be converted to proba)
                # but we just care about the order here.
                results[i]["rerank_score"] = float(rerank_scores[i])
            
            # Sort by rerank score descending
            results.sort(key=lambda x: x["rerank_score"], reverse=True)
            
            # Keep only the final TOP_K
            results = results[:FINAL_TOP_K]

        return results


def build_and_test():
    """Build index and run a sample search."""
    engine = SemanticSearch()
    engine.build_index()

    print("\n" + "=" * 60)
    print("  SEMANTIC SEARCH TEST")
    print("=" * 60)

    test_queries = [
        "My package hasn't arrived yet",
        "I can't reset my password",
        "Your app crashed and I lost my data",
        "I was charged twice for the same item",
        "How do I cancel my subscription?",
    ]

    for query in test_queries:
        print(f"\n  Query: \"{query}\"")
        results = engine.search(query, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"    [{i}] (score: {r['similarity_score']:.3f}) "
                  f"[{r['category']}]")
            print(f"        Issue: {r['customer_text'][:80]}...")
            print(f"        Solution: {r['company_response'][:80]}...")


if __name__ == "__main__":
    build_and_test()
