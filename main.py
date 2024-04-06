import zmq
import numpy as np
from datetime import datetime
from joblib import load

# Load the pre-trained model
model_with_scaler = load('model_with_scaler.h5')

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

# while True:
#     # Receive the date from the Unity client
#     date_str = socket.recv_string()
#     year, month, day = map(int, date_str.split(','))

#     # Predict the weather for the received date
#     future_weather = model_with_scaler.predict([[year, month, day]])

#     # Send the prediction back to the Unity client
#     socket.send_string(str(future_weather[0]))
# Send a date to predict weather for
date_str = "2024,4,6"  # Format: year, month, day
socket.send_string(date_str)

year, month, day = map(int, date_str.split(','))

    # Predict the weather for the received date
future_weather = model_with_scaler.predict([[year, month, day]])

# Receive the prediction
prediction = socket.recv_string()
print("Predicted weather:", prediction)