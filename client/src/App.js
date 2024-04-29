import React, { useState, useEffect } from 'react';
import { sendRequest } from './socket';

function App() {
  const [predictions, setPredictions] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await sendRequest('predict');
        const data = JSON.parse(response);
        setPredictions(data);
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();

    return () => {
      // Clean up resources if needed
    };
  }, []);

  return (
    <div>
      <h1>React Front End</h1>
      <h2>Predicted Weather:</h2>
      {predictions && (
        <ul>
          {Object.entries(predictions).map(([key, value]) => (
            <li key={key}>
              <strong>{key}:</strong> {value}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default App;
