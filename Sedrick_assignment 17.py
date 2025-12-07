import requests
import pandas as pd
import json
from datetime import datetime
import os

# API Configuration
api_key = '569171af82a7b752d1cc2a2686e3422b'

# List of cities to fetch weather data for
cities = [
    'Texas,US',
    'Tokyo,JP',
    'Sydney,AU',
    'Paris,FR',
    'Lagos,NG'
]


def fetch_weather_data(city_name, api_key):
    """
    Fetch weather data for a specific city

    Args:
        city_name (str): Name of the city
        api_key (str): OpenWeatherMap API key

    Returns:
        dict: Weather data or None if error
    """
    url = f'http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}'

    try:
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            print(f"Successfully fetched data for {city_name}")
            return data
        else:
            print(f"Error fetching weather data for {city_name}. Status code: {response.status_code}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Request failed for {city_name}: {e}")
        return None


def create_dataframe(weather_data, city_name):
    """
    Convert weather data to pandas DataFrame

    Args:
        weather_data (dict): Weather data from API
        city_name (str): Name of the city

    Returns:
        pd.DataFrame: Formatted weather data
    """
    if weather_data:
        df = pd.json_normalize(weather_data)
        df['city_query'] = city_name
        df['fetch_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return df
    return None


def save_individual_city_data(df, city_name):
    """
    Save individual city data to CSV

    Args:
        df (pd.DataFrame): Weather data DataFrame
        city_name (str): Name of the city
    """
    if df is not None:
        # Clean city name for filename
        clean_city_name = city_name.replace(',', '_').replace(' ', '_')
        filename = f"weather_data_{clean_city_name}.csv"
        df.to_csv(filename, index=False)
        print(f"Saved data to {filename}")


def main():
    """
    Main function to fetch weather data for all cities
    """
    print("=" * 60)
    print("WEATHER DATA COLLECTION REPORT")
    print("=" * 60)
    print(f"Collection started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"API Key: {api_key[:8]}..." + "*" * 20)
    print()

    all_dataframes = []
    city_results = {}

    # Process each city
    for i, city in enumerate(cities, 1):
        print(f"[{i}/5] Processing {city}...")

        # Fetch weather data
        weather_data = fetch_weather_data(city, api_key)

        if weather_data:
            # Create DataFrame
            df = create_dataframe(weather_data, city)

            # Save individual city data
            save_individual_city_data(df, city)

            # Store for combined analysis
            all_dataframes.append(df)
            city_results[city] = {
                'status': 'Success',
                'data': weather_data,
                'dataframe': df
            }

            # Display key information
            temp_kelvin = weather_data['main']['temp']
            temp_celsius = temp_kelvin - 273.15
            print(f"Temperature: {temp_celsius:.1f}°C")
            print(f"Weather: {weather_data['weather'][0]['description']}")
            print(f"Humidity: {weather_data['main']['humidity']}%")

        else:
            city_results[city] = {
                'status': 'Failed',
                'data': None,
                'dataframe': None
            }

        print()

    # Create combined dataset
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        combined_df.to_csv('combined_weather_data.csv', index=False)
        print("Combined dataset saved as 'combined_weather_data.csv'")

    # Generate summary report
    print("=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)

    successful_cities = [city for city, result in city_results.items() if result['status'] == 'Success']
    failed_cities = [city for city, result in city_results.items() if result['status'] == 'Failed']

    print(f"Successful requests: {len(successful_cities)}")
    print(f"Failed requests: {len(failed_cities)}")

    if successful_cities:
        print(f"\nSuccessful cities: {', '.join(successful_cities)}")

    if failed_cities:
        print(f"\nFailed cities: {', '.join(failed_cities)}")

    # Display sample of combined data
    if all_dataframes:
        print(f"\nSample of collected data:")
        print("-" * 40)
        sample_columns = ['name', 'main.temp', 'main.humidity', 'weather', 'city_query']
        available_columns = [col for col in sample_columns if col in combined_df.columns]
        print(combined_df[available_columns].head())

    print(f"\nCollection completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return city_results, all_dataframes


# Individual city scripts (as requested)
def fetch_single_city(city_name, label):
    """
    Fetch data for a single city with clear labeling
    """
    print(f"\n{'=' * 50}")
    print(f"SCRIPT FOR: {label}")
    print(f"{'=' * 50}")

    # API request
    url = f'http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}'
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        print(f"Data successfully fetched for {city_name}")

        # Convert to DataFrame
        df = pd.json_normalize(data)
        print(f"\nDataFrame for {label}:")
        print("-" * 30)
        print(df)

        # Save to file
        filename = f"{label.replace(' ', '_').replace(',', '')}_weather_data.csv"
        df.to_csv(filename, index=False)
        print(f"\nData saved to: {filename}")

        return df
    else:
        print(f"Error fetching data for {city_name}. Status code: {response.status_code}")
        return None


if __name__ == "__main__":
    # Run main collection
    results, dataframes = main()

    print("\n" + "=" * 60)
    print("INDIVIDUAL CITY SCRIPTS")
    print("=" * 60)

    # Individual city processing with clear labels
    city_labels = {
        'Texas,US': 'TEXAS, USA',
        'Tokyo,JP': 'TOKYO, JAPAN',
        'Sydney,AU': 'SYDNEY, AUSTRALIA',
        'Paris,FR': 'PARIS, FRANCE',
        'Lagos,NG': 'LAGOS, NIGERIA'
    }

    for city, label in city_labels.items():
        fetch_single_city(city, label)