# src/onnx_utils.py
import onnxruntime as ort
import numpy as np
from tokenizers import Tokenizer
from langchain_core.embeddings import Embeddings
from typing import List

class OnnxBgeEmbeddings(Embeddings): 
    """
    Lightweight wrapper to run BGE-Small using ONNX Runtime.
    No PyTorch required.
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        
        # Load Tokenizer
        tokenizer_path = f"{model_path}/tokenizer.json"
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=512)

        # Load ONNX Session (The Engine)
        # We assume the model file is named 'model.onnx' (standard export name)
        self.session = ort.InferenceSession(f"{model_path}/model.onnx")

    def _embed(self, texts: List[str]) -> List[List[float]]:
        # 1. Tokenize
        encodings = self.tokenizer.encode_batch(texts)
        
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)

        # 2. Run Inference
        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        }
        
        # Output is usually the first output node
        outputs = self.session.run(None, ort_inputs)
        last_hidden_state = outputs[0]

        # 3. CLS Pooling (Get the vector for the whole sentence)
        # BGE uses the first token (CLS) for embedding
        embeddings = last_hidden_state[:, 0, :]
        
        # 4. Normalize embeddings
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norm
        
        return embeddings.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]