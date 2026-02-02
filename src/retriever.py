import time
from pathlib import Path
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings # Required for the wrapper
from sentence_transformers import CrossEncoder, SentenceTransformer

BASE = Path(__file__).resolve().parent.parent

# --- NEW: Wrapper to make CLIP compatible with LangChain ---
class CLIPEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()

class MaritimeHybridRetriever:
    def __init__(self):
        # 1. Text-Only Embedding (BGE)
        self.text_emb = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )

        # 2. Multimodal Embedding (CLIP) - Wrapped for LangChain compatibility
        print("Loading CLIP for Multimodal retrieval...")
        self.clip_model = CLIPEmbeddings('clip-ViT-B-32')

        # 3. Reranker
        print("Loading Reranker...")
        self.reranker = CrossEncoder("BAAI/bge-reranker-base", activation_fn=None)

        # 4. Load Stores
        self.text_db = FAISS.load_local(
            str(BASE / "data" / "faiss" / "text"),
            self.text_emb,
            allow_dangerous_deserialization=True
        )

        # Now loading the image_db with the wrapped CLIP model
        self.image_db = FAISS.load_local(
            str(BASE / "data" / "faiss" / "images"),
            self.clip_model,
            allow_dangerous_deserialization=True
        )

    def retrieve(self, query: str, top_k_final: int = 5):
        timings = {}

        visual_triggers = {"image", "visual", "cover", "figure", "diagram", "photo", "illustration"}
        need_images = any(w in query.lower() for w in visual_triggers)

        # --- STAGE 1: TEXT RETRIEVAL (BGE) ---
        t0 = time.time()
        text_docs = self.text_db.similarity_search(query, k=15)
        timings["text_retrieval_time"] = time.time() - t0

        # --- STAGE 2: IMAGE RETRIEVAL (CLIP) ---
        image_docs = []
        if need_images:
            t1 = time.time()
            # This now uses the CLIP wrapper to search pixel vectors
            imgs = self.image_db.similarity_search(query, k=10)
            
            # Context Filtering (Matches images to pages found in text search)
            pages = {d.metadata.get("page_num") for d in text_docs}
            image_docs = [img for img in imgs if img.metadata.get("page_num") in pages]
            timings["image_retrieval_time"] = time.time() - t1

        initial_docs = text_docs + image_docs
        if not initial_docs: return [], timings

        # --- STAGE 3: RERANKING ---
        t2 = time.time()
        # Comparing query against text chunks AND image captions
        pairs = [[query, doc.page_content] for doc in initial_docs]
        scores = self.reranker.predict(pairs)
        
        scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
        
        final_docs = []
        for doc, score in scored_docs[:top_k_final]:
            doc.metadata["relevance_score"] = float(score)
            final_docs.append(doc)

        timings["reranking_time"] = time.time() - t2
        return final_docs, timings