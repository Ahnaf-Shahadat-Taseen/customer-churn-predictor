# app.py (FINAL - ORDER CORRECTED)

from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# --- MODEL LOADING ---
try:
    artifacts = joblib.load("churn_model_artifacts.joblib")
    model = artifacts['model']
    scaler = artifacts['scaler']
    model_columns = artifacts['model_columns']
    print("--- Model and artifacts loaded successfully. ---")
except Exception as e:
    print(f"FATAL: Error loading model artifacts: {e}")
    model = None

# --- WEB PAGES ---
@app.route('/')
def home():
    """Render the home page with the input form."""
    return render_template('index.html')

# --- API ENDPOINT ---
@app.route('/predict', methods=['POST'])
def predict():
    """Handle the prediction request and validate the input."""
    if model is None:
        return jsonify({'error': 'Model is not loaded. Please check server logs.'}), 500

    try:
        # Get the JSON data sent from the form
        data = request.get_json(force=True)

        # Convert the incoming JSON into a pandas DataFrame
        input_df = pd.DataFrame([data])
        
        # Force correct data types on the server side
        input_df['SeniorCitizen'] = pd.to_numeric(input_df['SeniorCitizen'])
        input_df['tenure'] = pd.to_numeric(input_df['tenure'])
        input_df['MonthlyCharges'] = pd.to_numeric(input_df['MonthlyCharges'])
        input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'])

        # One-hot encode categorical features
        input_df_encoded = pd.get_dummies(input_df)

        # Align columns with the training data to ensure consistency
        aligned_df = input_df_encoded.reindex(columns=model_columns, fill_value=0)
        
        # --- THE FINAL FIX ---
        # The scaler must receive columns in the EXACT same order it was trained on.
        # Based on the original data, SeniorCitizen came before tenure.
        numerical_cols_to_scale = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
        
        # Scale the numerical features using the loaded scaler
        aligned_df[numerical_cols_to_scale] = scaler.transform(aligned_df[numerical_cols_to_scale])
        
        # --- PREDICTION ---
        prediction_proba = model.predict_proba(aligned_df)[:, 1]
        churn_probability = float(prediction_proba[0])
        churn_result = "Yes" if churn_probability > 0.5 else "No"

        # Return the final result
        print("--- Prediction Successful! ---")
        return jsonify({
            'churn_prediction': churn_result,
            'churn_probability': round(churn_probability, 4)
        })

    except Exception as e:
        # Log the detailed error to the server console for debugging
        print(f"\n---!!! AN ERROR OCCURRED !!!---")
        import traceback
        traceback.print_exc()
        print(f"---!!! END OF ERROR !!!---\n")
        return jsonify({'error': 'An unexpected error occurred. Check the server logs for details.'}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=False, port=5000)