import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from langchain_community.embeddings import HuggingFaceEmbeddings
from src.retriever import MaritimeHybridRetriever
from src.generator import MaritimeGenerator

# ------------------------------
# Setup
# ------------------------------
emb = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

retriever = MaritimeHybridRetriever()
generator = MaritimeGenerator()

questions = [
    "What is the main focus of the Amrit Kaal Vision 2047?",
    "Who provided messages in the document, and what is the key message from the Prime Minister?",
    "Describe the cover image of the document.",
    "What are the 11 key themes identified in the Amrit Kaal Vision 2047?",
    "What visual elements are in the Executive Summary page?",
]

# ------------------------------
# Metric functions
# ------------------------------
def embed(texts):
    return np.array(emb.embed_documents(texts))

def context_precision(contexts, answer, threshold=0.6):
    if not contexts:
        return 0.0

    ctx_emb = embed(contexts)
    ans_emb = np.array(emb.embed_query(answer)).reshape(1, -1)

    sims = cosine_similarity(ctx_emb, ans_emb).flatten()
    relevant = sum(sim > threshold for sim in sims)

    return round(relevant / len(contexts), 3)

def context_recall(contexts, answer):
    if not contexts or not answer:
        return 0.0

    ctx_emb = embed(contexts)
    ans_emb = np.array(emb.embed_query(answer)).reshape(1, -1)

    sims = cosine_similarity(ans_emb, ctx_emb).flatten()
    return round(float(np.max(sims)), 3)

# ------------------------------
# Evaluation
# ------------------------------
results = []

for q in questions:
    docs, _ = retriever.retrieve(q)
    answer, _ = generator.generate(q, docs)

    contexts = [d.page_content for d in docs]

    results.append({
        "Question": q,
        "Context Precision": context_precision(contexts, answer),
        "Context Recall": context_recall(contexts, answer),
    })

print("\n📊 SEMANTIC RAG EVALUATION RESULTS")
for r in results:
    print(r)
