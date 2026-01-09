import os
import time
from pathlib import Path
from dotenv import load_dotenv
from llama_index.llms.groq import Groq

BASE = Path(__file__).resolve().parent.parent
load_dotenv(BASE / ".env")

class MaritimeGenerator:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        self.llm = Groq(
            model="llama-3.3-70b-versatile",
            api_key=api_key,
            temperature=0.1
        )

    def generate(self, query: str, docs: list):
        if not docs:
            return "Information not available in the document.", 0.0

        context = []
        for d in docs:
            page = d.metadata.get("page_num", "Unknown")
            image = d.metadata.get("image")

            if image:
                context.append(f"[Visual | Page {page} | {image}]: {d.page_content}")
            else:
                context.append(f"[Text | Page {page}]: {d.page_content}")

        prompt = f"""
Use ONLY the context below.

CONTEXT:
{chr(10).join(context)}

Question: {query}
Answer:
"""

        #  LLM RESPONSE TIME
        t0 = time.time()
        response = self.llm.complete(prompt)
        llm_time = time.time() - t0

        return str(response), llm_time
