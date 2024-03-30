import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# Load historical weather data from text file
def load_data(file_path):
    with open(file_path, 'r') as file:
        # Skip the header
        next(file)
        data = []
        for line in file:
            data_point = line.strip().split()  # Assuming space-separated values
            data.append(data_point)
    return data

# Load historical weather data
historical_data = load_data('historical_weather_data.txt')  # Replace 'historical_weather_data.txt' with your file path

# Convert data into a pandas DataFrame
historical_data_df = pd.DataFrame(historical_data, columns=['Date', 'Temperature'])  # Assuming date and temperature columns

# Preprocess data
historical_data_df['Date'] = pd.to_datetime(historical_data_df['Date'], format='%Y-%m-%d')  # Specify date format

historical_data_df['Year'] = historical_data_df['Date'].dt.year
historical_data_df['Month'] = historical_data_df['Date'].dt.month
historical_data_df['Day'] = historical_data_df['Date'].dt.day

# Select features and target
X = historical_data_df[['Year', 'Month', 'Day']]
y = historical_data_df['Temperature']  # Assuming temperature is the target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Predict future weather
future_date = datetime.now() + timedelta(days=7)  # Predicting weather for the next 7 days
future_year = future_date.year
future_month = future_date.month
future_day = future_date.day

future_weather = model.predict(scaler.transform([[future_year, future_month, future_day]]))
print(f'Predicted temperature for {future_date.date()}: {future_weather[0]}')
