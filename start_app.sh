#!/bin/bash

# Deactivate any active conda environment
conda deactivate 2>/dev/null

# Check if the virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment 'venv' not found. Creating and installing dependencies..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    echo "Virtual environment 'venv' found. Activating..."
    source venv/bin/activate
fi

# Run the Streamlit application
streamlit run app.py
