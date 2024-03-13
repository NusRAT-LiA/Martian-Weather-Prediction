import requests
import json
import pandas as pd

api_url = "https://cab.inta-csic.es/rems/wp-content/plugins/marsweather-widget/api.php"

def fetch_data():
    try:
        response = requests.get(api_url)
        data = response.json()
        return data
    except Exception as e:
        print("Error fetching data:", e)
        return None

# Process the fetched data
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

data = fetch_data()
processed_data = process_data(data)

# Print processed data 
# if processed_data:
#     print("Processed data:", json.dumps(processed_data, indent=4))

def prepare_data(processed_data):
    df = pd.DataFrame(processed_data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

prepared_data = prepare_data(processed_data)
print("Prepared data:")
print(prepared_data)

# output-
# Prepared data:
#              sol  min_temp  max_temp  pressure humidity wind_speed atmo_opacity sunrise sunset   uv_index  min_gts_temp  max_gts_temp
# date                                                                                                                                 
# 2024-03-06  4117       -69         4       791       --         --        Sunny   05:19  17:28   Moderate           -74            15
# 2024-03-05  4116       -67         9       788       --         --        Sunny   05:19  17:28   Moderate           -82            15
# 2024-03-04  4115       -71         8       781       --         --        Sunny   05:19  17:27   Moderate           -85            15
# 2024-03-03  4114       -68         6       783       --         --        Sunny   05:19  17:27   Moderate           -75            16
# 2024-03-02  4113       -72         7       778       --         --        Sunny   05:19  17:27   Moderate           -76            15
# ...          ...       ...       ...       ...      ...        ...          ...     ...    ...        ...           ...           ...
# 2012-08-18    12       -76       -18       741       --         --        Sunny   05:28  17:21  Very_High           -82             8
# 2012-08-17    11       -76       -11       740       --         --        Sunny   05:28  17:21  Very_High           -83             9
# 2012-08-16    10       -75       -16       739       --         --        Sunny   05:28  17:22  Very_High           -83             8
# 2012-08-15     9         0         0         0       --         --        Sunny   05:28  17:22         --             0             0
# 2012-08-07     1         0         0         0       --         --        Sunny   05:30  17:22         --             0             0

# [3893 rows x 12 columns]