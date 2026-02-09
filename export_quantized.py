# export_quantized.py
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from pathlib import Path

def export_tiny_model():
    model_id = "BAAI/bge-small-en-v1.5"
    save_path = Path("models/bge-onnx")
    
    # Load and export with quantization
    # 'arm64' or 'avx512' are common, but 'all' is safest for Azure
    model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
    
    # This is the magic line that shrinks the file
    from onnxruntime.quantization import quantize_dynamic, QuantType
    
    model.save_pretrained(save_path)
    
    # Quantize the specific model file
    quantize_dynamic(
        model_input=save_path / "model.onnx",
        model_output=save_path / "model_quantized.onnx",
        weight_type=QuantType.QUInt8
    )
    print("✅ Quantized model created!")

if __name__ == "__main__":
    export_tiny_model()