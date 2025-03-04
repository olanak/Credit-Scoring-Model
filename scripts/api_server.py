# Import libraries
from flask import Flask, request, jsonify
import joblib
import gdown
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load the model and scaler from Google Drive
MODEL_URL = "https://drive.google.com/uc?id=1IsZ3ed0tGfjnK_wnrSFyrFaX_bN9Msoc"
SCALER_URL = "https://drive.google.com/uc?id=1TV29ZdUREXSB3rnX1mxbiLeKnV1H9BTj"

# Define paths for model and scaler
model_path = "best_random_forest_model.pkl"
scaler_path = "scaler.pkl"

# Download the model and scaler if not already present
if not os.path.exists(model_path):
    gdown.download(MODEL_URL, model_path, quiet=False)

if not os.path.exists(scaler_path):
    gdown.download(SCALER_URL, scaler_path, quiet=False)

# Load the model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Check model and scaler types for debugging
print(f"Model type: {type(model)}")
print(f"Scaler type: {type(scaler)}")

# Root route to confirm API is live
@app.route('/', methods=['GET'])
def home():
    return "Credit Scoring API is live! Use /predict to make predictions."

# Prediction endpoint
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

        # Check for missing features in the input data
        missing_features = [feat for feat in numeric_features if feat not in input_df.columns]
        if missing_features:
            return jsonify({'error': f'Missing required features: {missing_features}'}), 400

        # Debugging - print input data before scaling
        print(f"Input data before scaling:\n{input_df[numeric_features]}")

        # Scale numeric features using the scaler
        if scaler:
            try:
                # Ensure the input data is a 2D array
                input_df[numeric_features] = scaler.transform(input_df[numeric_features].values)
            except Exception as e:
                return jsonify({'error': f'Scaling error: {str(e)}'}), 400

        # Debugging - print input data after scaling
        print(f"Input data after scaling:\n{input_df[numeric_features]}")

        # Make predictions using the model
        try:
            predictions = model.predict_proba(input_df)[:, 1]  # Probability of high-risk (class 1)
        except Exception as e:
            return jsonify({'error': f'Model prediction error: {str(e)}'}), 400

        # Return predictions as JSON
        return jsonify({'risk_probability': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    # Run the Flask app on the specified port
    PORT = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=PORT)
