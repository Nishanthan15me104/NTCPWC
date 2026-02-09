# export_to_onnx.py
import shutil
from pathlib import Path
from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTModelForCustomTasks
from transformers import AutoTokenizer, AutoProcessor

# Define Paths
BASE = Path(__file__).resolve().parent
MODELS_DIR = BASE / "models"
MODELS_DIR.mkdir(exist_ok=True)

def export_embedding_model():
    print("🚀 Exporting Text Embedding Model (BGE-Small)...")
    model_id = "BAAI/bge-small-en-v1.5"
    save_path = MODELS_DIR / "bge-onnx"
    
    # Export to ONNX using Optimum (wrapper for Transformers -> ONNX)
    model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"✅ Saved to {save_path}")

def export_clip_model():
    print("🚀 Exporting CLIP Model (Vision)...")
    model_id = "openai/clip-vit-base-patch32"
    save_path = MODELS_DIR / "clip-onnx"
    
    # CLIP is complex; usually for retrieval we just need the text encoder part 
    # to embed the user's query.
    # However, exporting full CLIP is tricky. 
    # STRATEGY: For the Free Tier, we will export ONLY the Text Encoder of CLIP.
    # This allows us to embed the user's query text to match image vectors.
    
    try:
        model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
        processor = AutoProcessor.from_pretrained(model_id)
        
        model.save_pretrained(save_path)
        processor.save_pretrained(save_path)
        print(f"✅ Saved to {save_path}")
    except Exception as e:
        print(f"⚠️ CLIP Export skipped (Complex): {e}")
        print("For Free Tier, we might disable Image Retrieval if this fails.")

if __name__ == "__main__":
    # You need to install 'optimum' and 'onnx' locally to run this script:
    # pip install optimum[onnxruntime]
    export_embedding_model()
    # export_clip_model() # Uncomment if you really need image search, but it's heavy.