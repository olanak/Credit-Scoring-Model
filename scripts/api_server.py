# Import libraries
from flask import Flask, request, jsonify
import joblib
import gdown
import os
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Google Drive URLs for model and scaler
MODEL_URL = "https://drive.google.com/uc?id=1TV29ZdUREXSB3rnX1mxbiLeKnV1H9BTj"  # Replace with your model file ID
SCALER_URL = "https://drive.google.com/uc?id=1IsZ3ed0tGfjnK_wnrSFyrFaX_bN9Msoc"  # Replace with your scaler file ID

# Paths to store downloaded files
MODEL_PATH = "best_random_forest_model.pkl"
SCALER_PATH = "scaler.pkl"

# Function to download a file if it doesn't exist
def download_file(url, path):
    if not os.path.exists(path):
        print(f"Downloading {path}...")
        try:
            gdown.download(url, path, quiet=False)
            print(f"Downloaded {path} successfully!")
        except Exception as e:
            print(f"Error downloading {path}: {str(e)}")
            exit(1)  # Stop execution if download fails

# Download model and scaler if not present
download_file(MODEL_URL, MODEL_PATH)
download_file(SCALER_URL, SCALER_PATH)

# Load the model and scaler
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and scaler loaded successfully!")
except Exception as e:
    print(f"Error loading model/scaler: {str(e)}")
    exit(1)

# Define prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        input_data = request.json

        # Convert input data to DataFrame
        input_df = pd.DataFrame(input_data)

        # Identify numeric features for scaling
        numeric_features = [
            'TotalTransactionAmount', 'AverageTransactionAmount', 'TransactionCount',
            'StdDevTransactionAmount', 'TransactionYear'
        ]

        # Ensure all required features exist in the input data
        missing_features = [feat for feat in numeric_features if feat not in input_df.columns]
        if missing_features:
            return jsonify({'error': f'Missing required features: {missing_features}'}), 400

        # Scale numeric features
        if scaler:
            input_df[numeric_features] = scaler.transform(input_df[numeric_features])

        # Make predictions
        predictions = model.predict_proba(input_df)[:, 1]  # Probability of high-risk (class 1)

        # Return predictions as JSON
        return jsonify({'risk_probability': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    # Run the Flask app on Render's port or locally
    PORT = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=PORT)
