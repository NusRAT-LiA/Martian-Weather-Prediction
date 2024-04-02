from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense

from config import API_URL

pd.set_option('future.no_silent_downcasting', True)

# Function to fetch data from API
def fetch_data():
    try:
        response = requests.get(API_URL)
        data = response.json()
        return data
    except Exception as e:
        print("Error fetching data:", e)
        return None

# Function to process data
def process_data(data):
    if not data:
        return None
    processed_data = []
    for sol in data.get('soles', []):
        processed_sol = {
            'date': sol.get('terrestrial_date', None),
            'sol': int(sol.get('sol', 0)),
            'min_temp': int(sol.get('min_temp', 0) if sol.get('min_temp') != '--' else 0),
            'max_temp': int(sol.get('max_temp', 0) if sol.get('max_temp') != '--' else 0),
            'pressure': int(sol.get('pressure', 0) if sol.get('pressure') != '--' else 0),
            'humidity': sol.get('abs_humidity', None),
            'wind_speed': sol.get('wind_speed', None),
            'atmo_opacity': sol.get('atmo_opacity', None),
            'sunrise': sol.get('sunrise', None),
            'sunset': sol.get('sunset', None),
            'uv_index': sol.get('local_uv_irradiance_index', None),
            'min_gts_temp': int(sol.get('min_gts_temp', 0) if sol.get('min_gts_temp') != '--' else 0),
            'max_gts_temp': int(sol.get('max_gts_temp', 0) if sol.get('max_gts_temp') != '--' else 0)
        }
        processed_data.append(processed_sol)
    return processed_data

# Function to prepare data
def prepare_data(processed_data):
    df = pd.DataFrame(processed_data)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year  # Add year column
    df['month'] = df['date'].dt.month  # Add month column
    df['day'] = df['date'].dt.day  # Add day column
    df.set_index('date', inplace=True)
    return df


# Function to preprocess data
def preprocess_data(prepared_data):
    # Your preprocessing steps here
    return prepared_data

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

# Function to create LSTM model
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to train LSTM model
def train_lstm_model(X_train, y_train, epochs=10, batch_size=32):
    model = create_lstm_model(X_train.shape[1:])
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model

# Function to predict future weather using LSTM model
def predict_future_weather(model, X):
    future_weather = model.predict(X)
    return future_weather[-1][0]

# Function to predict future weather for a column using LSTM
def predict_future_weather_for_column(column_to_predict):
    data = fetch_data()
    processed_data = process_data(data)
    prepared_data = prepare_data(processed_data)
    prepared_data = preprocess_data(prepared_data)

    # Reshape data for LSTM
    seq_length = 0  # Adjust as needed
    X = prepared_data[['year', 'month', 'day']].values
    y = prepared_data[[column_to_predict]].values

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Train LSTM model
    model = train_lstm_model(X_train, y_train)

    # Predict future weather
    future_date = datetime.now() 
    future_weather = predict_future_weather(model, X_test)

    print(f'Predicted {column_to_predict} for {future_date.date()}: {future_weather}')

# List of columns to predict
columns_to_predict = ["min_temp", "max_temp", "pressure", "min_gts_temp", "max_gts_temp", "uv_index_encoded"]

# Call the function for each column
for column in columns_to_predict:
    predict_future_weather_for_column(column)
