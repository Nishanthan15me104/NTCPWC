#!/bin/bash
# 1. Update pip and install requirements
python -m pip install --upgrade pip
pip install -r requirements.txt

# 2. Start Streamlit on the port Azure expects (8000)
# We add CORS/XSRF flags to prevent the "Connection Error" in the browser
python -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false