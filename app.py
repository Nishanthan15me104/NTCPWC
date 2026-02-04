import streamlit as st
import time
from src.retriever import MaritimeHybridRetriever
from src.generator import MaritimeGenerator

# 1. Initialize logic (cached so it doesn't reload on every click)
@st.cache_resource
def load_rag_system():
    # Toggle this to False if Azure crashes with "Out of Memory"
    USE_IMAGES = False 
    
    return MaritimeHybridRetriever(use_images=USE_IMAGES), MaritimeGenerator()

retriever, generator = load_rag_system()

st.title("🚢 Maritime RAG System")
st.markdown("Query the Vision 2047 Document with Hybrid Retrieval")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask about shore-side power or LNG bunkering..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing maritime documents..."):
            # Retrieval & Generation
            docs, timings = retriever.retrieve(prompt)
            answer, llm_time = generator.generate(prompt, docs)
            
            st.markdown(answer)
            
            # Show Metrics in an expander
            with st.expander("🔍 Retrieval Metrics"):
                st.write(f"Total Time: {llm_time:.2f}s")
                st.json(timings)

    st.session_state.messages.append({"role": "assistant", "content": answer})