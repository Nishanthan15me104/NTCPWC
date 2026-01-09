import time
from src.retriever import MaritimeHybridRetriever
from src.generator import MaritimeGenerator

retriever = MaritimeHybridRetriever()
generator = MaritimeGenerator()

metrics = []

queries = [
    "What is the main focus of the Amrit Kaal Vision 2047?",
    "Who provided messages in the document, and what is the key message from the Prime Minister?",
    "Describe the cover image of the document.",
    "What are the 11 key themes identified in the Amrit Kaal Vision 2047?",
    "What visual elements are in the Executive Summary page, and how do they relate to the content?"
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
