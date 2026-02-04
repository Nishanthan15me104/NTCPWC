import time
from pathlib import Path
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings 
from sentence_transformers import CrossEncoder, SentenceTransformer

BASE = Path(__file__).resolve().parent.parent

class CLIPEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()

class MaritimeHybridRetriever:
    def __init__(self, use_images=True):
        self.use_images = use_images
        
        # 1. Text-Only Embedding (Always Load)
        self.text_emb = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )

        # 2. Reranker (Always Load - but it's smaller than CLIP)
        print("Loading Reranker...")
        self.reranker = CrossEncoder("BAAI/bge-reranker-base", activation_fn=None)

        # 3. Load Text DB (Always Load)
        self.text_db = FAISS.load_local(
            str(BASE / "data" / "faiss" / "text"),
            self.text_emb,
            allow_dangerous_deserialization=True
        )

        # 4. Conditional Loading for Multimodal (The Memory Saver)
        if self.use_images:
            print("📸 Loading CLIP for Multimodal retrieval...")
            self.clip_model = CLIPEmbeddings('clip-ViT-B-32')
            
            print("📸 Loading Image DB...")
            self.image_db = FAISS.load_local(
                str(BASE / "data" / "faiss" / "images"),
                self.clip_model,
                allow_dangerous_deserialization=True
            )
        else:
            print("📝 Text-Only Mode: Skipping CLIP and Image DB to save RAM.")
            self.clip_model = None
            self.image_db = None

    def retrieve(self, query: str, top_k_final: int = 5):
        timings = {}

        # --- STAGE 1: TEXT RETRIEVAL ---
        t0 = time.time()
        text_docs = self.text_db.similarity_search(query, k=15)
        timings["text_retrieval_time"] = time.time() - t0

        # --- STAGE 2: IMAGE RETRIEVAL (Conditional) ---
        image_docs = []
        # Only search images if the model is loaded AND the query is visual
        visual_triggers = {"image", "visual", "cover", "figure", "diagram", "photo", "illustration"}
        need_images = any(w in query.lower() for w in visual_triggers)

        if self.use_images and need_images and self.image_db:
            t1 = time.time()
            imgs = self.image_db.similarity_search(query, k=10)
            
            # Context Filtering
            pages = {d.metadata.get("page_num") for d in text_docs}
            image_docs = [img for img in imgs if img.metadata.get("page_num") in pages]
            timings["image_retrieval_time"] = time.time() - t1

        initial_docs = text_docs + image_docs
        if not initial_docs: return [], timings

        # --- STAGE 3: RERANKING ---
        t2 = time.time()
        pairs = [[query, doc.page_content] for doc in initial_docs]
        scores = self.reranker.predict(pairs)
        
        scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
        
        final_docs = []
        for doc, score in scored_docs[:top_k_final]:
            doc.metadata["relevance_score"] = float(score)
            final_docs.append(doc)

        timings["reranking_time"] = time.time() - t2
        return final_docs, timings