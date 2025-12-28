from flask import Flask, render_template, request, jsonify
import joblib
import os
import sys
from src.model_preprocess import TextPreprocessor
from src.logger import logging
from src.exception import CustomException

app = Flask(__name__)

# Load Model
MODEL_PATH = os.path.join("artifacts", "best_spam_pipeline.pkl")

model = None

def load_model():
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            logging.info("Model loaded successfully")
        else:
            logging.warning("Model not found at startup. Please train the model first.")
    except Exception as e:
        raise CustomException(e, sys)

load_model()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        global model
        if model is None:
            # Try loading again just in case it was trained after app start
            load_model()
            if model is None:
                return jsonify({"error": "Model not available"}), 500

        data = request.get_json()
        message = data.get("message", "")
        
        if not message:
            return jsonify({"error": "Empty message"}), 400

        # Preprocess
        preprocessor = TextPreprocessor()
        clean_text = preprocessor.preprocess(message)
        
        # Predict
        prediction = model.predict([clean_text])[0]
        
        return jsonify({"prediction": prediction})

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
