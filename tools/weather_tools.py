# First, make sure you have the required libraries installed:
# pip install openmeteo-requests requests-cache retry-requests numpy pandas pytz

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import traceback

def get_comprehensive_agri_report(latitude: float, longitude: float, forecast_days: int = 7) -> dict:
    """
    Fetches a comprehensive weather and soil report for agricultural purposes.

    This function provides:
    1. Current weather conditions.
    2. A daily forecast for the specified number of days.
    3. Key hourly data for the next 48 hours for short-term planning.

    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        forecast_days (int): Number of days for the daily forecast (default is 7).

    Returns:
        dict: A dictionary containing 'current_weather' (dict),
              'daily_forecast' (pandas.DataFrame), and 'hourly_forecast' (pandas.DataFrame).
              Returns an error dictionary if the API call fails.
    """
    try:
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        url = "https://api.open-meteo.com/v1/forecast"

        # Define the variables for each type of forecast.
        current_vars = ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_gusts_10m", "wind_speed_10m", "rain", "showers"]
        hourly_vars = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation_probability", "et0_fao_evapotranspiration", "vapour_pressure_deficit", "soil_temperature_0cm", "soil_temperature_6cm", "soil_moisture_3_to_9cm", "soil_moisture_9_to_27cm", "weather_code", "rain", "showers", "precipitation"]
        daily_vars = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "precipitation_probability_max", "et0_fao_evapotranspiration", "shortwave_radiation_sum", "daylight_duration", "showers_sum", "rain_sum"]

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": current_vars,
            "hourly": hourly_vars,
            "daily": daily_vars,
            "timezone": "auto",
            "forecast_days": forecast_days,
            "forecast_hours": 48
        }
        
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]

        # --- Process Current Weather ---
        current = response.Current()
        current_weather = { "time": pd.to_datetime(current.Time(), unit="s", utc=True) }
        for i, var in enumerate(current_vars):
            current_weather[var] = current.Variables(i).Value()

        # --- Process Hourly Data ---
        hourly = response.Hourly()
        hourly_data = {"date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )}
        for i, var in enumerate(hourly_vars):
            hourly_data[var] = hourly.Variables(i).ValuesAsNumpy()
        hourly_dataframe = pd.DataFrame(data=hourly_data).set_index("date")

        # --- Process Daily Data ---
        daily = response.Daily()
        daily_data = {"date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        )}
        for i, var in enumerate(daily_vars):
            daily_data[var] = daily.Variables(i).ValuesAsNumpy()
        daily_dataframe = pd.DataFrame(data=daily_data).set_index("date")
        
        # ** THE FIX IS HERE **
        # Use response.Timezone() for the robust IANA name (e.g., 'Asia/Kolkata')
        local_tz_str = response.Timezone().decode("utf-8")
        
        current_weather["time"] = current_weather["time"].tz_convert(local_tz_str)
        hourly_dataframe.index = hourly_dataframe.index.tz_convert(local_tz_str)

        return {
            "location": {"latitude": latitude, "longitude": longitude, "elevation": response.Elevation(), "timezone": local_tz_str},
            "current_weather": current_weather,
            "hourly_forecast_48h": hourly_dataframe,
            "daily_forecast": daily_dataframe
        }
    except Exception:
        error_details = traceback.format_exc()
        return {"error": f"An unexpected error occurred:\n{error_details}"}

# --- Main execution block to run and test the function ---
if __name__ == "__main__":
    kanpur_lat = 26.4499
    kanpur_lon = 80.3319

    print(f"🛰️  Fetching comprehensive agricultural weather report for Kanpur...")
    agri_report = get_comprehensive_agri_report(latitude=kanpur_lat, longitude=kanpur_lon)

    if "error" in agri_report:
        print(f"❌ {agri_report['error']}")
    else:
        print("\n" + "="*50)
        print("✅ Report Fetched Successfully!")
        print(f"📍 Location: {agri_report['location']}")
        print("="*50 + "\n")

        print("--- 🌡️  Current Weather ---")
        current = agri_report['current_weather']
        print(f"Time: {current['time'].strftime('%Y-%m-%d %I:%M %p')}")
        print(f"Temperature: {current['temperature_2m']:.2f}°C")
        print(f"Humidity: {current['relative_humidity_2m']:.1f}%")
        print("\n" + "="*50 + "\n")
        
        print(f"--- 📅 Daily Forecast ---")
        print(agri_report['daily_forecast'])
        print("\n" + "="*50 + "\n")

        print("---💧 Hourly Forecast ---")
        print(agri_report['hourly_forecast_48h'])