import zmq

# Setup ZeroMQ context and socket
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

# Send a request to the server
socket.send_string("predict")

# Receive the predictions from the server
predictions = socket.recv_json()

# Print the received predictions
print("Predictions:")
for column, value in predictions.items():
    print(f"{column}: {value}")
