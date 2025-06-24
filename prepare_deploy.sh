#!/bin/bash
# Deployment preparation script for Fintel

echo "===== Fintel Deployment Preparation ====="
echo "This script will help you prepare for deploying your Fintel app."

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "Error: Git is not installed. Please install Git first."
    exit 1
fi

# Check if git repo is initialized
if [ ! -d ".git" ]; then
    echo "Initializing Git repository..."
    git init
fi

# Create necessary directories if they don't exist
mkdir -p data
mkdir -p models

# Check if model files exist
if [ ! -f "models/xgboost_model.pkl" ] || [ ! -f "models/scaler.pkl" ] || [ ! -f "models/feature_names.pkl" ] || [ ! -f "models/label_mapping.pkl" ]; then
    echo "Warning: Model files are missing. Make sure to train your model before deployment."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting."
        exit 1
    fi
fi

# Check for data files
data_files=$(ls data/*.csv 2>/dev/null | wc -l)
if [ "$data_files" -eq 0 ]; then
    echo "Warning: No data files found in the data directory."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting."
        exit 1
    fi
fi

# Ensure requirements.txt is up-to-date
echo "Verifying requirements.txt..."
pip freeze > requirements_current.txt
diff requirements.txt requirements_current.txt > /dev/null
if [ $? -ne 0 ]; then
    echo "Warning: Your requirements.txt may not be up-to-date with your environment."
    read -p "Update requirements.txt with current environment? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        mv requirements_current.txt requirements.txt
        echo "requirements.txt updated."
    else
        rm requirements_current.txt
    fi
else
    rm requirements_current.txt
    echo "requirements.txt is up-to-date."
fi

echo "===== Deployment Preparation Complete ====="
echo ""
echo "To deploy on Streamlit Cloud:"
echo "1. Push your code to GitHub"
echo "2. Go to https://streamlit.io/cloud"
echo "3. Connect your GitHub repo"
echo "4. Point to streamlit_app.py"
echo ""
echo "To test locally:"
echo "streamlit run streamlit_app.py"
