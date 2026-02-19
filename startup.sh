#!/bin/bash
# 1. Start the Brain
uvicorn api:app --host 127.0.0.1 --port 8001 &

# 2. Wait for Health (Wait up to 150 seconds)
MAX_RETRIES=30
while [ $COUNT -lt $MAX_RETRIES ]; do
  if curl -s http://127.0.0.1:8001/health | grep -q "healthy"; then
    echo "✅ Backend Ready!"
    break
  fi
  sleep 5
  COUNT=$((COUNT+1))
done

# 3. Start the Face
# Note: Use --server.address 0.0.0.0 so Azure can "see" it
python -m streamlit run ui.py --server.port 8000 --server.address 0.0.0.0 --server.enableCORS=false