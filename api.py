import logging
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware # Added for browser security
from pydantic import BaseModel, Field # Added Field for validation
from src.retriever import MaritimeHybridRetriever
from src.generator import MaritimeGenerator

# --- OBSERVABILITY ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("MaritimeAPI")

app = FastAPI(title="Maritime RAG API", version="1.0.0")

# --- SENIOR PILLAR: CORS Security ---
# Allows Streamlit to talk to FastAPI without browser blocks
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- STATE TRACKING ---
retriever = None
generator = None

@app.on_event("startup")
async def startup_event():
    global retriever, generator
    logger.info("🚀 Starting up Maritime API...")
    try:
        retriever = MaritimeHybridRetriever(use_images=False)
        generator = MaritimeGenerator()
        logger.info("✅ Models loaded successfully.")
    except Exception as e:
        logger.critical(f"🔥 Critical Failure loading models: {e}")
        raise e
    
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Shutting down Maritime API...")
    global retriever, generator
    del retriever
    del generator
    logger.info("Cleanup complete.")

class QueryRequest(BaseModel):
    # Added max_length to protect your Groq bill/credits
    prompt: str = Field(..., max_length=1000)

@app.post("/ask")
async def ask_maritime(request: QueryRequest):
    try:
        logger.info(f"📨 Received query: {request.prompt}")
        
        # Retrieval (Async hand-off to prevent freezing)
        docs, timings = await asyncio.to_thread(retriever.retrieve, request.prompt)
        
        # FIX: If no docs, return a STREAM of text so the UI doesn't crash
        if not docs:
            async def empty_stream():
                yield "I couldn't find any relevant information in the documents."
            return StreamingResponse(empty_stream(), media_type="text/plain")

        # Return the actual LLM stream
        return StreamingResponse(
            generator.generate_stream(request.prompt, docs), 
            media_type="text/plain"
        )

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Engine Error")

@app.get("/health")
def health_check():
    # Azure uses this to see if the "Engine" is actually running
    return {"status": "healthy", "engine_ready": retriever is not None}