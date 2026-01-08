import os
from dotenv import load_dotenv
from datasets import Dataset

from ragas import evaluate
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

from langchain_groq import ChatGroq

from src.retriever import MaritimeHybridRetriever
from src.generator import MaritimeGenerator


# --------------------------------------------------
# Load environment variables (.env in project root)
# --------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")


# --------------------------------------------------
# Use SAME LLM for generation + evaluation
# --------------------------------------------------
ragas_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0,
)

retriever = MaritimeHybridRetriever()
generator = MaritimeGenerator()


# --------------------------------------------------
# Evaluation Queries
# --------------------------------------------------
questions = [
    "What is the main focus of the Amrit Kaal Vision 2047?",
    "Who provided messages in the document, and what is the key message from the Prime Minister?",
    "Describe the cover image of the document.",
    "What are the 11 key themes identified in the Amrit Kaal Vision 2047?",
    "What visual elements are in the Executive Summary page?",
]


# --------------------------------------------------
# Build RAGAS Dataset
# --------------------------------------------------
data = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": [],
}

for q in questions:
    docs, _ = retriever.retrieve(q)
    answer, _ = generator.generate(q, docs)

    contexts = [d.page_content for d in docs]

    data["question"].append(q)
    data["answer"].append(answer)
    data["contexts"].append(contexts)

    # Proxy ground truth (official RAGAS recommendation)
    data["ground_truth"].append(contexts[0] if contexts else "")

dataset = Dataset.from_dict(data)


# --------------------------------------------------
# RAGAS Evaluation (✅ CORRECT)
# --------------------------------------------------
results = evaluate(
    dataset=dataset,
    metrics=[
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ],
)

print("\n📊 RAGAS EVALUATION RESULTS")
print(results)
