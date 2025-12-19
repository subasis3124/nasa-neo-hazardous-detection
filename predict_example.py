import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "id": 2085990,
    "name": "85990 (1999 JV6)",
    "absolute_magnitude_h": 20.27,
    "min_diameter_km": 0.234723,
    "max_diameter_km": 0.524856,
    "close_approach_date": "2015-01-05",
    "close_approach_date_full": "2015-Jan-05 11:16",
    "epoch_date_close_approach": 1420456560000,
    "relative_velocity_kps": 7.694743,
    "miss_distance_km": 1.246329e+07,
    "orbiting_body": "Earth",
    "is_sentry_object": False,
    "nasa_jpl_url": "https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr=2085990"
}
response = requests.post(url, json=data)
print(response.json())
