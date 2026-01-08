import re
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# -----------------------------
# Paths
# -----------------------------
BASE = Path(__file__).resolve().parent.parent
IMG_DIR = BASE / "data" / "extracted" / "images"
OUT = BASE / "data" / "faiss" / "images"


# -----------------------------
# Device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Load BLIP captioning model
# -----------------------------
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)

model.eval()


# -----------------------------
# Create Documents
# -----------------------------
docs = []

for img_path in IMG_DIR.glob("*.png"):
    # Expected filename: img_<page>_<n>.png
    match = re.search(r"img_(\d+)_\d+", img_path.name)
    if not match:
        continue

    page_num = int(match.group(1))

    image = Image.open(img_path).convert("RGB")

    with torch.no_grad():
        inputs = processor(image, return_tensors="pt").to(device)
        output = model.generate(**inputs)
        caption = processor.decode(
            output[0],
            skip_special_tokens=True
        )

    docs.append(
        Document(
            page_content=caption,
            metadata={
                "page_num": page_num,
                "image": img_path.name
            }
        )
    )


# -----------------------------
# Build FAISS index
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

db = FAISS.from_documents(docs, embeddings)

OUT.mkdir(parents=True, exist_ok=True)
db.save_local(str(OUT))


print("✅ Image FAISS index created")
