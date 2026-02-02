import re
from pathlib import Path
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer # Use this for CLIP

# -----------------------------
# Paths & Device
# -----------------------------
BASE = Path(__file__).resolve().parent.parent
IMG_DIR = BASE / "data" / "extracted" / "images"
OUT = BASE / "data" / "faiss" / "images"
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# 1. Load BLIP (For Captioning - LLM Context)
# -----------------------------
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)
blip_model.eval()

# -----------------------------
# 2. Load CLIP (For Multimodal Indexing)
# -----------------------------
# This model embeds images and text into the same vector space.
print("Loading CLIP model for pixel embedding...")
clip_model = SentenceTransformer('clip-ViT-B-32') 

# -----------------------------
# Create Documents with Pixel Embeddings
# -----------------------------
docs = []
image_vectors = []

for img_path in IMG_DIR.glob("*.png"):
    match = re.search(r"img_(\d+)_\d+", img_path.name)
    if not match: continue

    page_num = int(match.group(1))
    image = Image.open(img_path).convert("RGB")

    # A. Generate Text Caption (for Reranker and LLM)
    with torch.no_grad():
        inputs = processor(image, return_tensors="pt").to(device)
        output = blip_model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)

    # B. Generate Pixel Embedding (for Retrieval)
    # We encode the actual image pixels here
    img_emb = clip_model.encode(image)
    image_vectors.append(img_emb)

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
# Build FAISS Index using Image Vectors
# -----------------------------
# We manually zip the vectors and docs because LangChain's standard 
# HuggingFaceEmbeddings only supports text-to-vector.
text_embeddings = clip_model.encode(["placeholder"]) # Just to get dimension
vector_dim = len(image_vectors[0])

# We use a trick here: FAISS.from_embeddings allows us to provide the 
# pre-computed pixel vectors directly.
# We pass a lambda that can encode text queries later.
db = FAISS.from_embeddings(
    text_embeddings=list(zip([d.page_content for d in docs], image_vectors)),
    embedding=clip_model, # This will be used for text queries later
    metadatas=[d.metadata for d in docs]
)

OUT.mkdir(parents=True, exist_ok=True)
db.save_local(str(OUT))
print(f"✅ Multimodal Image FAISS index created with {len(docs)} images")