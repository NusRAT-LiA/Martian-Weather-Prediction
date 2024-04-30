from flask import Flask, jsonify
import joblib
from datetime import datetime
from model import ModelWithScaler
import subprocess
import zmq
import os

app = Flask(__name__)

# Function to execute the training process
def train_model():
    subprocess.run(["python3", "Train.py"])

def predict_future_weather(model_with_scaler, future_date):
    future_year = future_date.year
    future_month = future_date.month
    future_day = future_date.day

    # Predict weather using the provided model and future date
    future_weather = model_with_scaler.predict([[future_year, future_month, future_day]])
    return future_weather[0]

# List of columns to predict
columns_to_predict = ['min_temp', 'max_temp', 'pressure','uv_index_encoded','min_gts_temp','max_gts_temp']

# Directory where models are stored
models_dir = os.path.dirname(__file__)  

# Train the model before starting the server
train_model()

@app.route('/predict', methods=['GET'])
def predict():
    predictions = {}
    future_date = datetime.now()  
    for column in columns_to_predict:
        model_file_path = os.path.join(models_dir, f'{column}_model_with_scaler.h5')
        model_with_scaler = joblib.load(model_file_path)
        future_weather = predict_future_weather(model_with_scaler, future_date)
        predictions[column] = future_weather

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
