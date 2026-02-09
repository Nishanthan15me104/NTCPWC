import time
import os
from pathlib import Path
from langchain_community.vectorstores import FAISS

# Import our custom ONNX wrapper
from src.utils.onnx_utils import OnnxBgeEmbeddings

# Reranker import is commented out for Free Tier
# from sentence_transformers import CrossEncoder 

BASE = Path(__file__).resolve().parent.parent

class MaritimeHybridRetriever:
    def __init__(self, use_images=False):
        self.use_images = use_images
        # 1. Define Model Path
        self.model_path = BASE / "models" / "bge-onnx"
        
        # Safety Check: Ensure the user ran the export script
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"🚨 ONNX Model not found at {self.model_path}. "
                "Did you run 'export_to_onnx.py' locally first?"
            )

        print(f"🚀 Loading ONNX Embedding Model from {self.model_path}...")
        self.embedding_model = OnnxBgeEmbeddings(model_path=str(self.model_path))

        # 2. Load FAISS Index
        index_path = BASE / "data" / "faiss" / "text"
        if not index_path.exists():
             raise FileNotFoundError(f"🚨 FAISS Index not found at {index_path}.")

        print("📂 Loading FAISS Index...")
        self.text_db = FAISS.load_local(
            str(index_path),
            self.embedding_model,
            allow_dangerous_deserialization=True
        )

        # 3. Reranker (DISABLED for Free Tier)
        self.reranker = None
        # self.reranker = CrossEncoder("BAAI/bge-reranker-base")

    def retrieve(self, query: str, top_k: int = 5):
        timings = {}
        t0 = time.time()
        
        # 1. Similarity Search (Uses ONNX)
        # We fetch slightly more docs just in case, but return top_k
        docs = self.text_db.similarity_search(query, k=top_k)
        
        timings["retrieval_time"] = time.time() - t0
        
        # --- RERANKING (Disabled) ---
        # If you enable this later, you must uncomment the import above
        # and add 'sentence-transformers' to requirements.txt (Cloud).
        #
        # if self.reranker:
        #     pairs = [[query, doc.page_content] for doc in docs]
        #     scores = self.reranker.predict(pairs)
        #     ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        #     docs = [doc for doc, score in ranked]
        
        return docs, timings