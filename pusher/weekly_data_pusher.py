import pandas as pd
import requests
from src.utils.utils import fetch_data, create_engine_for_database
from sqlalchemy import create_engine
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL')
NASA_API_KEY = os.getenv('NASA_API_KEY')

user_name = os.getenv('POSTGRES_USER')
password = os.getenv('POSTGRES_PASSWORD')
name = os.getenv('POSTGRES_DB')
host = os.getenv('POSTGRES_HOST')
port = os.getenv('POSTGRES_PORT')


## fetching the data
def fetch_data(start_date, end_date, api):
    url = f"https://api.nasa.gov/neo/rest/v1/feed?start_date={start_date}&end_date={end_date}&api_key={api}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print('Unable to connect')
        return None


def extract_and_store_data_list():
    neo_list = []

    today = datetime.now().date()
    start_date = today - timedelta(days=6)
    end_date = start_date + timedelta(days=6)

    while start_date <= today:
        # Adjust end_date to not go beyond today
        if end_date > today:
            end_date = today

        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        print("-" * 50)
        print(f"Fetching data from [{start_str}] to [{end_str}]")
        print("-" * 50)

        # Fetch the data
        data = fetch_data(start_str, end_str, NASA_API_KEY)

        # Ensure data is valid before processing
        if data and "near_earth_objects" in data:
            for date, asteroids in data["near_earth_objects"].items():
                for asteroid in asteroids:
                    # loop over all close approaches, not just [0]
                    for ca_data in asteroid.get("close_approach_data", []):
                        neo_list.append({
                                "id": asteroid.get("id"),
                                "name": asteroid.get("name"),
                                "absolute_magnitude_h": asteroid.get("absolute_magnitude_h"),
                                "min_diameter_km": asteroid.get("estimated_diameter", {}).get("kilometers", {}).get("estimated_diameter_min"),
                                "max_diameter_km": asteroid.get("estimated_diameter", {}).get("kilometers", {}).get("estimated_diameter_max"),
                                "close_approach_date": ca_data.get("close_approach_date"),
                                "close_approach_date_full": ca_data.get("close_approach_date_full"),
                                "epoch_date_close_approach": ca_data.get("epoch_date_close_approach"),
                                "relative_velocity_kps": float(ca_data.get("relative_velocity", {}).get("kilometers_per_second", 0.0)),
                                "miss_distance_km": float(ca_data.get("miss_distance", {}).get("kilometers", 0.0)),
                                "orbiting_body": ca_data.get("orbiting_body"),
                                "is_potentially_hazardous": asteroid.get("is_potentially_hazardous_asteroid"),
                                "is_sentry_object": asteroid.get("is_sentry_object"),
                                "nasa_jpl_url": asteroid.get("nasa_jpl_url"),
                            })

        # Move to the next 7-day window
        start_date = end_date + timedelta(days=1)
        end_date = start_date + timedelta(days=6)

    ## converting the list into data frame after loop finishes
    neo_df = pd.DataFrame(neo_list)
    return neo_df


def push_to_pg(df, table_name):
    engine = create_engine_for_database(
        user_name=user_name, 
        password=password, 
        host=host, 
        port=port, 
        database_name=name
    )
    df.to_sql(table_name, con=engine, if_exists='append', index=False)


if __name__ == "__main__":
    print("[START] NEO Data Pipeline Execution")
    df = extract_and_store_data_list()
    if not df.empty:
        push_to_pg(df, 'train_neo')
        push_to_pg(df, 'test_neo')
        print(f"[SUCCESS] Inserted {len(df)} rows into train_neo and test_neo")
    else:
        print("[WARNING] No data fetched.")