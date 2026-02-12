from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = pickle.load(open("model/churn_model.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    features = np.array([[
        data['gender'],
        data['senior'],
        data['tenure'],
        data['monthly'],
        data['total']
    ]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    result = {
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "probability": round(float(probability), 2)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
