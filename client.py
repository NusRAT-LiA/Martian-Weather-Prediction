import zmq

# Setup ZeroMQ context and socket
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

# Send a request to the server
socket.send_string("predict")

# Receive the response from the server
response = socket.recv_string()
print("Server response:", response)
