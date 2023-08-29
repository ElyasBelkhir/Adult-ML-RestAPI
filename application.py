#Becker,Barry and Kohavi,Ronny. (1996). Adult. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20.

from flask import Flask, request, jsonify
import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import re

app = Flask(__name__)

# Load the pre-trained model
model = load('model.joblib')
scaler = load('scaler.joblib')
required_columns = load('columns.joblib')


@app.route('/predict', methods=['POST'])
def predict():
    print("Home route accessed")

    try:
        # Get JSON data from POST request

        if not request.json:
            return jsonify({"error": "No JSON payload provided"})
        
        data = request.json
        input_df = pd.DataFrame([data])

        print("Input DataFrame:")
        print(input_df)
        print(input_df.dtypes)
      

        # Perform one-hot encoding (you'll need to match this with how you one-hot encoded in the training phase)
        encoded_df = pd.get_dummies(input_df, columns=['workclass', 'education', 'marital_status',
                                 'occupation', 'relationship', 'race',
                                 'sex', 'native_country'], dtype=int)
        def add_space_after_second_underscore(text):
            return re.sub(r'(_[^_]*_)', r'\1 ', text)

        encoded_df.columns = [add_space_after_second_underscore(col) for col in encoded_df.columns]

        print("Encoded DataFrame:")
        print(encoded_df)

        missing_cols = [col for col in required_columns if col not in encoded_df.columns]
        missing_df = pd.DataFrame({col: [0] for col in missing_cols})
        encoded_df = pd.concat([encoded_df, missing_df], axis=1)

        # Ensure the order of columns matches the original dataset
        encoded_df = encoded_df[required_columns]
        print("After Check")
        print(encoded_df)
       

        # Scale the features
        scaled_data = scaler.transform(encoded_df[['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']])
        
        # Replace original columns with scaled ones
        encoded_df[['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']] = scaled_data

        print("Scaled Data:")
        print(scaled_data)

        # Make the prediction
        prediction = model.predict_proba(encoded_df)
        prediction1 = model.predict(encoded_df)

        print("Prediction:", prediction)
        print("Prediction:", prediction1)

        # Return the prediction
        return jsonify({"prediction": int(prediction1[0])})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
