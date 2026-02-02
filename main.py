import time
from src.retriever import MaritimeHybridRetriever
from src.generator import MaritimeGenerator

retriever = MaritimeHybridRetriever()
generator = MaritimeGenerator()

metrics = []

queries = [
       "Existing shore-side power supplies in the world applied for different terminal type",
        "what is LNG-Bunkering?",
        "Tell me about Green Belt"
]

for i, query in enumerate(queries, 1):
    start = time.time()

    docs, retriever_times = retriever.retrieve(query)
    answer, llm_time = generator.generate(query, docs)

    total_time = time.time() - start

    metrics.append({
        "Query": i,
        "Text Retrieval (s)": round(retriever_times.get("text_retrieval_time", 0), 3),
        "Image Retrieval (s)": round(retriever_times.get("image_retrieval_time", 0), 3),
        "LLM Response (s)": round(llm_time, 3),
        "Total Time (s)": round(total_time, 3)
    })

    print(f"\nQ{i}: {query}")
    print(answer)

print("\n PERFORMANCE METRICS")
for m in metrics:
    print(m)
