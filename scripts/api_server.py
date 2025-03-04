# Import libraries
from flask import Flask, request, jsonify
import joblib
import os
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Define paths for Render Disk storage
PERSISTENT_PATH = "/persistent"
model_path = os.path.join(PERSISTENT_PATH, "best_random_forest_model.pkl")
scaler_path = os.path.join(PERSISTENT_PATH, "scaler.pkl")

# Check if model and scaler exist
if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    raise FileNotFoundError("Model or scaler file not found in Render Disk. Upload them first!")

# Load the model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Define prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        input_data = request.json

        # Convert input to DataFrame
        input_df = pd.DataFrame(input_data)

        # Required numeric features
        numeric_features = [
            'TotalTransactionAmount', 'AverageTransactionAmount', 'TransactionCount',
            'StdDevTransactionAmount', 'TransactionYear'
        ]

        # Check for missing features
        missing_features = [feat for feat in numeric_features if feat not in input_df.columns]
        if missing_features:
            return jsonify({'error': f'Missing required features: {missing_features}'}), 400

        # Scale numeric features
        input_df[numeric_features] = scaler.transform(input_df[numeric_features])

        # Predict risk probability
        predictions = model.predict_proba(input_df)[:, 1]  

        # Return predictions
        return jsonify({'risk_probability': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    PORT = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=PORT)
