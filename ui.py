# --- ui.py ---
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8001/ask"

st.set_page_config(page_title="Maritime RAG", page_icon="🚢")
st.title("🚢 Maritime RAG System")
st.markdown("Query the Vision 2047 Document (Streaming Enabled)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about LNG bunkering..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # We use an empty container to fill the text as it streams
        response_placeholder = st.empty()
        full_response = ""

        try:
            # 1. Open a streaming connection
            with requests.post(API_URL, json={"prompt": prompt}, stream=True, timeout=60) as r:
                if r.status_code == 200:
                    # 2. Iterate over chunks of text coming from the API
                    for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            full_response += chunk
                            # 3. Update the UI in real-time
                            response_placeholder.markdown(full_response + "▌")
                    
                    # Final clean look without the cursor
                    response_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    st.error(f"Server Error: {r.status_code}")
                    
        except requests.exceptions.ConnectionError:
            st.error("🚨 Connection failed. Is the API running on port 8001?")