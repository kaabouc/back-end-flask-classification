from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import joblib
import pandas as pd

# Create Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app, origins=["http://localhost:3000"])  # Allow requests from React app

# Load the trained model
model = joblib.load('random_forest_car_evaluation (2).pkl')

# Preprocess function to convert input data to match model format
def preprocess_input(data):
    # Create a DataFrame from the input data
    df = pd.DataFrame([data])
    
    # Encode categorical variables in the same way as during training
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # Get the feature names used during model training
    model_feature_names = model.feature_names_in_  # Use the feature names from the model
    model_columns = pd.Index(model_feature_names)
    
    # Reindex the DataFrame to have the same columns as the model
    df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)
    
    return df_encoded

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Ensure that all required fields are present in the request
        required_fields = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Invalid input data. Missing required fields."}), 400
        
        # Preprocess the input
        input_data = preprocess_input(data)
        
        # Make the prediction
        prediction = model.predict(input_data)
        
        # Interpret the prediction result
        result = 'good' if prediction[0] == 1 else 'bad'
        
        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Start the Flask server
if __name__ == '__main__':
    app.run(debug=True)
