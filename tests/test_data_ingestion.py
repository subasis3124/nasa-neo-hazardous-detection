from unittest.mock import patch
import pandas as pd
from src.custom.data_ingestion import DataIngestion

# Sample DataFrame (from your example)
sample_df = pd.DataFrame({
    "id": [2085990, 2523915],
    "name": ["85990 (1999 JV6)", "523915 (1997 VM4)"],
    "absolute_magnitude_h": [20.27, 18.57],
    "min_diameter_km": [0.234723, 0.513517],
    "max_diameter_km": [0.524856, 1.148259],
    "close_approach_date": ["2015-01-05", "2015-01-05"],
    "close_approach_date_full": ["2015-Jan-05 11:16", "2015-Jan-05 11:56"],
    "epoch_date_close_approach": [1420456560000, 1420458960000],
    "relative_velocity_kps": [7.694743, 16.169622],
    "miss_distance_km": [1.246329e+07, 5.815821e+07],
    "orbiting_body": ["Earth", "Earth"],
    "is_potentially_hazardous": [True, False],
    "is_sentry_object": [False, False],
    "nasa_jpl_url": [
        "https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr=2085990",
        "https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr=2523915"
    ]
})


@patch("src.custom.data_ingestion.read_data_from_pg")
@patch("pandas.DataFrame.to_csv")
def test_initiate_data_ingestion(mock_to_csv, mock_read_pg):
    # Mock the database read function to return sample dataframe
    mock_read_pg.return_value = sample_df

    ingestion = DataIngestion()
    result = ingestion.initiate_data_ingestion(
        password="pass",
        username="user",
        host="localhost",
        port=5432,
        name="db"
    )

    # Check if database read function is called correctly
    mock_read_pg.assert_called_once_with(
        "user", "pass", "localhost", 5432, "db", ingestion.table_name
    )

    # Check if DataFrame.to_csv was called
    mock_to_csv.assert_called_once()

    # Ensure the returned dataframe equals our sample
    pd.testing.assert_frame_equal(result, sample_df)
