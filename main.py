from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the model
model = tf.keras.models.load_model("heart_disease_modell.h5")

# Initialize Flask app
app = Flask(__name__)

# Initialize StandardScaler
scaler = StandardScaler()

# Define a route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    features = scaler.transform(features)  # Scale the input

    prediction = model.predict(features)
    result = "Positive" if prediction[0] > 0.5 else "Negative"
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the model
model = tf.keras.models.load_model("heart_disease_modell.h5")

# Initialize Flask app
app = Flask(__name__)

# Initialize StandardScaler
scaler = StandardScaler()

# Define a route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    features = scaler.transform(features)  # Scale the input

    prediction = model.predict(features)
    result = "Positive" if prediction[0] > 0.5 else "Negative"
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
