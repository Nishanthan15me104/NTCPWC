#!/bin/bash
# 1. Ensure we are in the correct directory
# Oryx extracts to /tmp or /home/site/wwwroot. 
# This command finds your code regardless of where Azure put it.
cd "$(dirname "$0")"

echo "🚀 Starting the Brain (FastAPI)..."
# Using 'python -m uvicorn' ensures it uses the virtual environment's python
python -m uvicorn api:app --host 127.0.0.1 --port 8001 &

# 2. Wait for Health (Wait up to 150 seconds)
MAX_RETRIES=30
COUNT=0
echo "⏳ Waiting for Backend to be healthy..."
while [ $COUNT -lt $MAX_RETRIES ]; do
  if curl -s http://127.0.0.1:8001/health | grep -q "healthy"; then
    echo "✅ Backend Ready!"
    break
  fi
  sleep 5
  COUNT=$((COUNT+1))
done

# 3. Start the Face (Streamlit)
echo "🚢 Starting the Maritime UI..."
python -m streamlit run ui.py --server.port 8000 --server.address 0.0.0.0 --server.enableCORS=false