## import Library
import requests
import pandas as pd

# key # place

api_key = '569171af82a7b752d1cc2a2686e3422b'
city = 'London'

#Building the api request
url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}'
response = requests.get(url)

## Make a request to obtain the data.
if response.status_code == 200:
    data = response.json() # Convert JSON reponse to python dictionnary
    #print(data)
else:
    print(f"Error fetching weather data. Status code: {response.status_code}")


#if data:
#    df = pd.json_normalize(data)
#    print(df)
#else:

#Convert to a pandas dataframe.
df = pd.json_normalize(data)
print(df)