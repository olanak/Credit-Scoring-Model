# Import libraries
from flask import Flask, request, jsonify
import joblib
import gdown
import os
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the model and scaler from Google Drive
MODEL_URL = "https://drive.google.com/uc?id=1TV29ZdUREXSB3rnX1mxbiLeKnV1H9BTj"  # Replace with your model file ID
SCALER_URL = "https://drive.google.com/uc?id=1IsZ3ed0tGfjnK_wnrSFyrFaX_bN9Msoc"  # Replace with your scaler file ID

# Download the model and scaler if not already present
model_path = "../models/best_random_forest_model.pkl"
scaler_path = "../models/scaler.pkl"

if not os.path.exists(model_path):
    gdown.download(MODEL_URL, model_path, quiet=False)

if not os.path.exists(scaler_path):
    gdown.download(SCALER_URL, scaler_path, quiet=False)

# Load the model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

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