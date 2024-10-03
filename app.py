# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import joblib
# import pandas as pd

# # Initialize the Flask application
# app = Flask(__name__)
# CORS(app, origins=["http://127.0.0.1:3000 "])
# # Load your pre-trained Random Forest model
# model = joblib.load('models/random_forest_model.pkl')

# # Define mappings for categorical variables (same as used in training)
# buying_mapping = {"vhigh": 0, "high": 1, "med": 2, "low": 3}
# maint_mapping = {"vhigh": 0, "high": 1, "med": 2, "low": 3}
# doors_mapping = {"2": 0, "3": 1, "4": 2, "5more": 3}
# persons_mapping = {"more": 0, "2": 1}  # Assuming "more" is for more than 2 persons
# lug_boot_mapping = {"small": 0, "med": 1, "big": 2}
# safety_mapping = {"low": 0, "med": 1, "high": 2}

# # Define the prediction route

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get the input data from the request in JSON format
#     data = request.get_json()
    
#     # Encode the input data using the mappings
#     try:
#         encoded_data = {
#             "buying": buying_mapping[data['buying']],
#             "maint": maint_mapping[data['maint']],
#             "doors": doors_mapping[data['doors']],
#             "persons": persons_mapping[data['persons']],
#             "lug_boot": lug_boot_mapping[data['lug_boot']],
#             "safety": safety_mapping[data['safety']],
#         }

#         # Convert the encoded data to a DataFrame for model input
#         input_data = pd.DataFrame(encoded_data, index=[0])
        
#         # Make a prediction using the loaded model
#         prediction = model.predict(input_data)
        
#         # Define result based on prediction (assuming 1 means "good")
#         result = "good" if prediction[0] == 1 else "bad"
        
#         # Return the result as a JSON response
#         return jsonify({'result': result})
    
#     except KeyError as e:
#         return jsonify({'error': f'Missing data for: {str(e)}'}), 400
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# # Run the app
# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)

# Enable CORS for all routes with credentials support
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)

# Load your trained Random Forest model
model = joblib.load('./models/random_forest_model.pkl')

# Label mappings for input
label_mappings = {
    'buying': {'vhigh': 3, 'high': 2, 'med': 1, 'low': 0},
    'maint': {'vhigh': 3, 'high': 2, 'med': 1, 'low': 0},
    'doors': {'2': 0, '3': 1, '4': 2, '5more': 3},
    'persons': {'2': 0, '4': 1, 'more': 2},
    'lug_boot': {'small': 0, 'med': 1, 'big': 2},
    'safety': {'low': 0, 'med': 1, 'high': 2}
}

# Label mappings for output
class_mappings = {
    0: 'good',  # unaccepted
    1: 'acc',    # acceptable
    2: 'unacc',   # good
    3: 'vgood'   # very good
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        input_data = [
            label_mappings['buying'][data['buying']],
            label_mappings['maint'][data['maint']],
            label_mappings['doors'][data['doors']],
            label_mappings['persons'][data['persons']],
            label_mappings['lug_boot'][data['lug_boot']],
            label_mappings['safety'][data['safety']]
        ]
    except KeyError as e:
        return jsonify({'error': f'Invalid value for {str(e)}'}), 400

    prediction = model.predict([input_data])
    prediction_class = int(prediction[0])
    
    # Map numeric prediction to a readable class label
    predicted_label = class_mappings.get(prediction_class, 'Unknown')

    return jsonify({'prediction': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
