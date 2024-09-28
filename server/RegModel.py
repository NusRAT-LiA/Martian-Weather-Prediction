import requests
import json
import pandas as pd
import numpy as np
from config import API_URL
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

pd.set_option('future.no_silent_downcasting', True)

reg = Ridge(alpha=0.1)

def fetchData():
    try:
        response = requests.get(API_URL)
        data = response.json()
        return data
    except Exception as e:
        print("Error fetching data:", e)
        return None

def processData(data):
    if not data:
        return None

    processedData = []
    for sol in data.get('soles', []):
        processedSol = {
            'date': sol.get('terrestrial_date', None),
            'sol': int(sol.get('sol', 0)),
            'minTemp': int(sol.get('min_temp', 0) if sol.get('min_temp') != '--' else 0),
            'maxTemp': int(sol.get('max_temp', 0) if sol.get('max_temp') != '--' else 0),
            'pressure': int(sol.get('pressure', 0) if sol.get('pressure') != '--' else 0),
            'humidity': sol.get('abs_humidity', None),
            'windSpeed': sol.get('wind_speed', None),
            'atmoOpacity': sol.get('atmo_opacity', None),
            'sunrise': sol.get('sunrise', None),
            'sunset': sol.get('sunset', None),
            'uvIndex': sol.get('local_uv_irradiance_index', None),
            'minGtsTemp': int(sol.get('min_gts_temp', 0) if sol.get('min_gts_temp') != '--' else 0),
            'maxGtsTemp': int(sol.get('max_gts_temp', 0) if sol.get('max_gts_temp') != '--' else 0)
        }
        processedData.append(processedSol)
    return processedData

data = fetchData()
processedData = processData(data)

def prepareData(processedData):
    df = pd.DataFrame(processedData)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

def createPredictions(preparedData):
    predictors = ["minTemp", "maxTemp", "pressure", 
                  "minGtsTemp", "maxGtsTemp", "uvIndexEncoded"]
    
    # Dictionary to store predictions for each target column
    allPredictions = {}
    
    # Iterate over each column (except the date column)
    for column in preparedData.columns:
        if column == 'date':
            continue
        
        targetColumn = column
        
        X = preparedData[predictors]
        y = preparedData[targetColumn]
        
        reg.fit(X, y) 
        
        predictions = reg.predict(X)
        
        error = mean_squared_error(y, predictions)
        
        print("Mean Squared Error for {}: {}".format(targetColumn, error))
        
        allPredictions[targetColumn] = predictions
    
    return allPredictions

def makePredictionsForNewData(newData, model, predictors):
    allPredictions = {}
    for column in newData.columns:
        if column == 'date':
            continue
        
        targetColumn = column
        
        XNew = newData[predictors]
        
        predictions = model.predict(XNew)
        
        allPredictions[targetColumn] = predictions
    
    return allPredictions

preparedData = prepareData(processedData)

# Replace '--' with NaN
preparedData = preparedData.replace('--', np.nan).infer_objects(copy=False)

# Calculate null percentages for each column
nullPercentage = (preparedData.isnull().sum() / len(preparedData)) * 100

# Remove columns with null percentage > 50
columnsToRemove = nullPercentage[nullPercentage > 50].index
preparedData.drop(columnsToRemove, axis=1, inplace=True)

# Fill missing values in 'uv_index' with mode
modeUvIndex = preparedData['uvIndex'].mode()[0]
preparedData['uvIndex'] = preparedData['uvIndex'].fillna(modeUvIndex)

# Fill missing values in 'atmo_opacity' with mode
mostFrequentAtmoOpacity = preparedData['atmoOpacity'].mode()[0]
preparedData['atmoOpacity'] = preparedData['atmoOpacity'].fillna(mostFrequentAtmoOpacity)

# Drop rows when rover wasnt activated
preparedData.drop(['2012-08-15', '2012-08-07'], inplace=True)

uvIndexMapping = {'Low': 0, 'Moderate': 1, 'High': 2, 'Very_High': 3}
preparedData['uvIndexEncoded'] = preparedData['uvIndex'].map(uvIndexMapping)

# One-Hot Encoding for 'atmoOpacity' column
atmoOpacityEncoded = pd.get_dummies(preparedData['atmoOpacity'], prefix='atmoOpacity')

preparedData = pd.concat([preparedData, atmoOpacityEncoded], axis=1)

# Drop the original non-numerical columns
preparedData.drop(['uvIndex', 'atmoOpacity', 'sunrise','sunset','sol'], axis=1, inplace=True)

print("updated PreparedData with time components:")
print(preparedData)

allPredictions = createPredictions(preparedData)
