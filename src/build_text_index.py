import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------
# Paths
# -----------------------------
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "extracted" / "metadata.json"
OUT = BASE / "data" / "faiss" / "text"


# -----------------------------
# Load metadata
# -----------------------------
with open(DATA, "r", encoding="utf-8") as f:
    metadata = json.load(f)

assert "text" in metadata, "metadata.json missing 'text' key"
pages = metadata["text"]


# -----------------------------
# Create Documents
# pages format:
# [
#   {"text": "...", "page_num": 1},
#   ...
# ]
# -----------------------------
docs = [
    Document(
        page_content=p["text"],
        metadata={"page_num": p["page_num"]}
    )
    for p in pages
    if isinstance(p, dict) and p.get("text")
]


# -----------------------------
# Split into chunks
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
)

docs = splitter.split_documents(docs)


# -----------------------------
# Build FAISS index
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

db = FAISS.from_documents(docs, embeddings)

OUT.mkdir(parents=True, exist_ok=True)
db.save_local(str(OUT))


print("✅ Text FAISS index created")
