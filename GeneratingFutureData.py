import requests
import json
import pandas as pd
import numpy as np
from config import API_URL
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import HistGradientBoostingRegressor
import pickle
from datetime import datetime
import yfinance as yf
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM


pd.set_option('future.no_silent_downcasting', True)
tf.random.set_seed(0)

# Function to fetch data from the API
def fetch_data():
    try:
        response = requests.get(API_URL)
        data = response.json()
        return data
    except Exception as e:
        print("Error fetching data:", e)
        return None

# Function to process the fetched data
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

# Fetch data
data = fetch_data()
# Process data
processed_data = process_data(data)

# Function to prepare the processed data into a DataFrame
def prepare_data(processed_data):
    df = pd.DataFrame(processed_data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

# Prepare data
prepared_data = prepare_data(processed_data)

# Replace '--' with NaN
prepared_data = prepared_data.replace('--', np.nan).infer_objects(copy=False)

# Calculate null percentages for each column
null_percentage = (prepared_data.isnull().sum() / len(prepared_data)) * 100

# Remove columns with null percentage > 50
columns_to_remove = null_percentage[null_percentage > 50].index
prepared_data.drop(columns_to_remove, axis=1, inplace=True)

# Fill missing values in 'uv_index' with mode
mode_uv_index = prepared_data['uv_index'].mode()[0]
prepared_data['uv_index'] = prepared_data['uv_index'].fillna(mode_uv_index)

# Fill missing values in 'atmo_opacity' with mode
most_frequent_atmo_opacity = prepared_data['atmo_opacity'].mode()[0]
prepared_data['atmo_opacity'] = prepared_data['atmo_opacity'].fillna(most_frequent_atmo_opacity)

# Drop specific rows
prepared_data.drop(['2012-08-15', '2012-08-07'], inplace=True)

# Label Encoding for 'uv_index' column
uv_index_mapping = {'Low': 0, 'Moderate': 1, 'High': 2, 'Very_High': 3}
prepared_data['uv_index_encoded'] = prepared_data['uv_index'].map(uv_index_mapping)

# One-Hot Encoding for 'atmo_opacity' column
atmo_opacity_encoded = pd.get_dummies(prepared_data['atmo_opacity'], prefix='atmo_opacity')

# Concatenate the encoded columns with the original dataframe
prepared_data= pd.concat([prepared_data, atmo_opacity_encoded], axis=1)

# Drop the original non-numerical columns
prepared_data.drop(['uv_index', 'atmo_opacity', 'sunrise','sunset','sol', 'atmo_opacity_Sunny'], axis=1, inplace=True)

# print the updated prepared data with time components
print("Updated Prepared data with time components:")
print(prepared_data)

# Generate the input and output sequences
n_lookback = 60  # length of input sequences (lookback period)
n_forecast = 30  # length of output sequences (forecast period)

X = []
Y = []

for i in range(n_lookback, len(prepared_data) - n_forecast + 1):
    X.append(prepared_data.values[i - n_lookback: i])
    Y.append(prepared_data.values[i: i + n_forecast])

X = np.array(X)
Y = np.array(Y)

# Fit the model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, prepared_data.shape[1])))
model.add(LSTM(units=50))
model.add(Dense(n_forecast))

model.compile(loss='mean_squared_error', optimizer='adam')
Y = Y.reshape(-1, n_forecast)

model.fit(X, Y, epochs=100, batch_size=32, verbose=0)

# Generate the forecasts
X_ = prepared_data.values[- n_lookback:]  # last available input sequence
X_ = X_.reshape(1, n_lookback, prepared_data.shape[1])

Y_ = model.predict(X_).reshape(-1, 1)

# Organize the results in a DataFrame
df_past = prepared_data.reset_index()
df_past.rename(columns={'index': 'Date'}, inplace=True)

df_future = pd.DataFrame(columns=['Date', 'Forecast'])
df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
df_future['Forecast'] = Y_.flatten()

results = df_past.append(df_future).set_index('Date')

# Plot the results
results.plot(title='Forecast')

# Save the trained model
with open('trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load the trained model
with open('trained_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Predict future data
future_dates = pd.date_range(start=prepared_data.index[-1] + pd.Timedelta(days=1), periods=n_forecast)
future_data = np.array([prepared_data.values[-n_lookback:]])
future_predictions = loaded_model.predict(future_data)

print("Future Predictions:")
print(future_predictions)