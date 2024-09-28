import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Define the API URL
apiUrl = "https://cab.inta-csic.es/rems/wp-content/plugins/marsweather-widget/api.php"

def fetchData():
    try:
        response = requests.get(apiUrl)
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

def prepareData(processedData):
    df = pd.DataFrame(processedData)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

data = fetchData()
processedData = processData(data)
preparedData = prepareData(processedData)

#replace '--' with NaN and infer objects for correct types
preparedData = preparedData.replace('--', np.nan).infer_objects(copy=False)

# calculate null percentages and remove columns with more than 50% missing data
nullPercentage = (preparedData.isnull().sum() / len(preparedData)) * 100
columnsToRemove = nullPercentage[nullPercentage > 50].index
preparedData.drop(columnsToRemove, axis=1, inplace=True)

# fill missing values in "uv_index" with mode
modeUvIndex = preparedData['uvIndex'].mode()[0]
preparedData['uvIndex'] = preparedData['uvIndex'].fillna(modeUvIndex)

# drop rows when rover didnt send any data
preparedData.drop(['2012-08-15', '2012-08-07'], inplace=True, errors='ignore')

uvIndexMapping = {'Low': 0, 'Moderate': 1, 'High': 2, 'Very_High': 3}
preparedData['uvIndexEncoded'] = preparedData['uvIndex'].map(uvIndexMapping)

 # drop the original non-numerical columns
preparedData.drop(['uvIndex', 'atmoOpacity', 'sunrise', 'sunset', 'sol'], axis=1, inplace=True)
# Drop rows where any value is 0
preparedData = preparedData[(preparedData != 0).all(axis=1)]

def prepareDataForLstm(preparedData, nSteps=10):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaledData = scaler.fit_transform(preparedData.values)
    
    X, y = [], []
    for i in range(nSteps, len(scaledData)):
        X.append(scaledData[i-nSteps:i])
        y.append(scaledData[i])
    X, y = np.array(X), np.array(y)
    
    return X, y, scaler

def buildLstmModel(inputShape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=inputShape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=inputShape[1]))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

nSteps = 10
X, y, scaler = prepareDataForLstm(preparedData, nSteps=nSteps)

trainSize = int(len(X) * 0.8)
XTrain, XTest = X[:trainSize], X[trainSize:]
yTrain, yTest = y[:trainSize], y[trainSize:]

model = buildLstmModel(inputShape=(XTrain.shape[1], XTrain.shape[2]))
model.fit(XTrain, yTrain, epochs=50, batch_size=32)

predictions = model.predict(XTest)
predictions = scaler.inverse_transform(predictions)
yTestRescaled = scaler.inverse_transform(yTest)

mse = mean_squared_error(yTestRescaled, predictions)
print(f"Mean Squared Error: {mse}")

# for i in range(10):
#     print(f"Actual: {yTestRescaled[i]}, Predicted: {predictions[i]}")

def getComplementaryColor(color):
    colorRgb = np.array(mcolors.to_rgb(color))
    hsv = mcolors.rgb_to_hsv(colorRgb)
    hsv[0] = (hsv[0] + 0.5) % 1.0
    complementaryRgb = mcolors.hsv_to_rgb(hsv)
    return mcolors.to_hex(complementaryRgb)

parameters = preparedData.columns
colorsActual = list(mcolors.TABLEAU_COLORS.values())

plt.figure(figsize=(14, len(parameters) * 3))

for i, param in enumerate(parameters):
    plt.subplot(len(parameters), 1, i + 1)
    
    actualColor = colorsActual[i % len(colorsActual)]
    predictedColor = getComplementaryColor(actualColor)
    
    plt.plot(yTestRescaled[:, i], label=f'Actual {param}', color=actualColor, alpha=0.6)
    plt.plot(predictions[:, i], label=f'Predicted {param}', color=predictedColor, linestyle='--', alpha=0.6)
    
    plt.title(f'Actual vs Predicted {param}')
    plt.xlabel('Time Steps')
    plt.ylabel(param)
    plt.legend()

plt.tight_layout()
plt.show()
