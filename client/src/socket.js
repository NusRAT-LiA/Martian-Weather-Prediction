// const zmq = require('zeromq');

// const createSocket = () => {
//   const socket = zmq.socket('req');
//   socket.connect('tcp://localhost:5555'); 

//   return socket;
// };

// const sendRequest = async (request) => {
//   const socket = createSocket();

//   const sendPromise = new Promise((resolve, reject) => {
//     socket.send(request, (err) => {
//       if (err) reject(err);
//       else resolve();
//     });
//   });

//   const receivePromise = new Promise((resolve, reject) => {
//     socket.on('message', (response) => {
//       resolve(response.toString());
//     });
//   });

//   await sendPromise; 
//   const response = await receivePromise; 

//   socket.close();

//   return response;
// };

// module.exports = { sendRequest };
