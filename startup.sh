#!/bin/bash
# 1. Ensure we are in the correct directory
cd "$(dirname "$0")"

# 2. Define the path to our virtual environment binaries
# This ensures we use exactly what was installed during deployment
VENV_BIN="./antenv/bin"

echo "🚀 Starting the Brain (FastAPI)..."
# Start uvicorn using the direct path
$VENV_BIN/uvicorn api:app --host 127.0.0.1 --port 8001 &

# 3. Wait for Health (Wait up to 150 seconds)
MAX_RETRIES=30
COUNT=0
echo "⏳ Waiting for Backend to be healthy..."
while [ $COUNT -lt $MAX_RETRIES ]; do
  # Check if the process is running on localhost
  if curl -s http://127.0.0.1:8001/health | grep -q "healthy"; then
    echo "✅ Backend Ready!"
    break
  fi
  echo "...still waiting ($COUNT/$MAX_RETRIES)..."
  sleep 5
  COUNT=$((COUNT+1))
done

# 4. Start the Face (Streamlit)
echo "🚢 Starting the Maritime UI..."
# Use the direct path to the streamlit binary
$VENV_BIN/python -m streamlit run ui.py --server.port 8000 --server.address 0.0.0.0 --server.enableCORS=false