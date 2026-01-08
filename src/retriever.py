import time
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

BASE = Path(__file__).resolve().parent.parent

class MaritimeHybridRetriever:
    def __init__(self):
        self.emb = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )

        self.text_db = FAISS.load_local(
            str(BASE / "data" / "faiss" / "text"),
            self.emb,
            allow_dangerous_deserialization=True
        )

        self.image_db = FAISS.load_local(
            str(BASE / "data" / "faiss" / "images"),
            self.emb,
            allow_dangerous_deserialization=True
        )

    def retrieve(self, query: str):
        timings = {}

        visual_triggers = {
            "image", "visual", "cover", "figure",
            "diagram", "photo", "illustration"
        }

        need_images = any(w in query.lower() for w in visual_triggers)

        # ⏱️ TEXT RETRIEVAL (includes embedding time)
        t0 = time.time()
        text_docs = self.text_db.similarity_search(query, k=5)
        timings["text_retrieval_time"] = time.time() - t0

        if not need_images:
            return text_docs, timings

        pages = {d.metadata.get("page_num") for d in text_docs if d.metadata.get("page_num")}

        # ⏱️ IMAGE RETRIEVAL
        t1 = time.time()
        imgs = self.image_db.similarity_search(query, k=10)
        timings["image_retrieval_time"] = time.time() - t1

        image_docs = [
            img for img in imgs
            if img.metadata.get("page_num") in pages
        ]

        if not image_docs:
            return [], timings

        return text_docs + image_docs, timings
