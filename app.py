from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend requests

# Load the trained ML model
with open("health_care_1.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.json
        features = np.array([
            float(data["DRG_Definition"]),
            float(data["Provider_Id"]),
            float(data["Total_Discharges"]),
            float(data["Average_Covered_Charges"]),
            float(data["Average_Total_Payments"]),
            float(data["Average_Medicare_Payments"]),
        ]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)
        return jsonify({"prediction": int(prediction[0])})  # Convert NumPy int to normal int

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=10000)  # Corrected to listen on all interfaces

