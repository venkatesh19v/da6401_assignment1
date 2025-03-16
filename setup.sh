#!/bin/bash

echo "Creating virtual environment..."
python3 -m venv venv

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Deactivate the virtual environment
deactivate

echo "Setup complete. Virtual environment created and dependencies installed."
