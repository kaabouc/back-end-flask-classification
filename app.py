from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)

# Load your pre-trained Random Forest model
model = joblib.load('models/random_forest_model.pkl')

# Define a mapping for categorical variables
buying_mapping = {"vhigh": 0, "high": 1, "med": 2, "low": 3}
maint_mapping = {"vhigh": 0, "high": 1, "med": 2, "low": 3}
doors_mapping = {"2": 0, "3": 1, "4": 2, "5more": 3}
persons_mapping = {"more": 0, "2": 1}  # Assuming "more" is for more than 2 persons
lug_boot_mapping = {"small": 0, "med": 1, "big": 2}
safety_mapping = {"low": 0, "med": 1, "high": 2}

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request in JSON format
    data = request.get_json()
    
    # Encode the input data
    encoded_data = {
        "buying": buying_mapping[data['buying']],
        "maint": maint_mapping[data['maint']],
        "doors": doors_mapping[data['doors']],
        "persons": persons_mapping[data['persons']],
        "lug_boot": lug_boot_mapping[data['lug_boot']],
        "safety": safety_mapping[data['safety']],
    }

    # Convert the encoded data to a DataFrame for model input
    input_data = pd.DataFrame(encoded_data, index=[0])
    
    # Make a prediction using the loaded model
    prediction = model.predict(input_data)
    
    # You can set a threshold for classification or directly return based on prediction
    result = "good" if prediction[0] == 1 else "bad"  # Assuming 1 means "good"
    
    # Return the result as a JSON response
    return jsonify({'result': result})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
