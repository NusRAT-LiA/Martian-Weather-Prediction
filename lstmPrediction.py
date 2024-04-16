import requests
import json
import pandas as pd
import numpy as np
from config import API_URL
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


pd.set_option('future.no_silent_downcasting', True)

# Define the Ridge regression model
reg = Ridge(alpha=0.1)

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


def create_predictions(prepared_data):
    
    # Define predictors
    predictors = ["min_temp", "max_temp", "pressure", 
                  "min_gts_temp", "max_gts_temp", "uv_index_encoded"]
    
    # Dictionary to store predictions for each target column
    all_predictions = {}
    
    # Iterate over each column (except the date column)
    for column in prepared_data.columns:
        if column == 'date':
            continue
        
        # Set the target column
        target_column = column
        
        # Split data into features (X) and target (y)
        X = prepared_data[predictors]
        y = prepared_data[target_column]
        
        # Fit the model
        reg.fit(X, y) 
        
        # Make predictions
        predictions = reg.predict(X)
        
        # Calculate mean squared error
        error = mean_squared_error(y, predictions)
        
        # Print mean squared error
        print("Mean Squared Error for {}: {}".format(target_column, error))
        
        # Store predictions for the target column
        all_predictions[target_column] = predictions
    
    return all_predictions

# Function to make predictions for new data
def make_predictions_for_new_data(new_data, model, predictors):

    # Iterate over each column (except the date column)
    all_predictions = {}
    for column in new_data.columns:
        if column == 'date':
            continue
        
        # Set the target column
        target_column = column
        
        # Get features (X) for new data
        X_new = new_data[predictors]
        
        # Make predictions
        predictions = model.predict(X_new)
        
        # Store predictions for the target column
        all_predictions[target_column] = predictions
    
    return all_predictions


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
prepared_data.drop(['uv_index', 'atmo_opacity', 'sunrise','sunset','sol'], axis=1, inplace=True)



# Print the updated prepared data with time components
print("Updated Prepared data with time components:")
print(prepared_data)

# Create predictions for each target column
all_predictions = create_predictions(prepared_data)

