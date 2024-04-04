import joblib
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from config import API_URL
import yfinance as yf

pd.set_option('future.no_silent_downcasting', True)

class ModelWithScaler():
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

def fetch_data():
    try:
        response = requests.get(API_URL)
        data = response.json()
        return data
    except Exception as e:
        print("Error fetching data:", e)
        return None

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

def prepare_data(processed_data):
    df = pd.DataFrame(processed_data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

def preprocess_data(prepared_data):
    prepared_data = prepared_data.replace('--', np.nan).infer_objects(copy=False)
    null_percentage = (prepared_data.isnull().sum() / len(prepared_data)) * 100
    columns_to_remove = null_percentage[null_percentage > 50].index
    prepared_data.drop(columns_to_remove, axis=1, inplace=True)

    mode_uv_index = prepared_data['uv_index'].mode()[0]
    prepared_data['uv_index'] = prepared_data['uv_index'].fillna(mode_uv_index)
    most_frequent_atmo_opacity = prepared_data['atmo_opacity'].mode()[0]
    prepared_data['atmo_opacity'] = prepared_data['atmo_opacity'].fillna(most_frequent_atmo_opacity)
    prepared_data.drop(['2012-08-15', '2012-08-07'], inplace=True)

    uv_index_mapping = {'Low': 0, 'Moderate': 1, 'High': 2, 'Very_High': 3}
    prepared_data['uv_index_encoded'] = prepared_data['uv_index'].map(uv_index_mapping)

    atmo_opacity_encoded = pd.get_dummies(prepared_data['atmo_opacity'], prefix='atmo_opacity')
    prepared_data= pd.concat([prepared_data, atmo_opacity_encoded], axis=1)

    prepared_data.drop(['uv_index', 'atmo_opacity', 'sunrise','sunset','sol', 'atmo_opacity_Sunny'], axis=1, inplace=True)

    prepared_data['year'] = prepared_data.index.year
    prepared_data['month'] = prepared_data.index.month
    prepared_data['day'] = prepared_data.index.day
    return prepared_data

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Combine model and scaler into one object
    model_with_scaler = ModelWithScaler(model, scaler)

    # Save the combined model and scaler using joblib
    joblib.dump(model_with_scaler, 'model_with_scaler.h5')

    return model_with_scaler, X_test_scaled, y_test

def predict_future_weather(model_with_scaler, future_date):
    future_year = future_date.year
    future_month = future_date.month
    future_day = future_date.day

    future_weather = model_with_scaler.predict([[future_year, future_month, future_day]])
    return future_weather[0]

def predict_future_weather_for_column(column_to_predict):
    data = fetch_data()
    processed_data = process_data(data)
    prepared_data = prepare_data(processed_data)
    prepared_data = preprocess_data(prepared_data)

    X = prepared_data[['year', 'month', 'day']]
    y = prepared_data[[column_to_predict]]

    model_with_scaler, _, _ = train_model(X, y)

    future_date = datetime.now() 
    future_weather = predict_future_weather(model_with_scaler, future_date)

    print(f'Predicted {column_to_predict} for {future_date.date()}: {future_weather}')

# List of columns to predict
columns_to_predict = ["min_temp", "max_temp", "pressure", "min_gts_temp", "max_gts_temp", "uv_index_encoded"]

# Call the function for each column
for column in columns_to_predict:
    predict_future_weather_for_column(column)
