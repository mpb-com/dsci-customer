import os

current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
ROOT_DIR = os.path.dirname(current_dir)

SQL_FOLDER = f"{ROOT_DIR}/sql/"
BQ_LOGGING_FOLDER = f"{ROOT_DIR}/logs/sql_history/"
LOGGING_FOLDER = f"{ROOT_DIR}/logs/"
TEST_FOLDER = f"{ROOT_DIR}/tests/"

GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "")
BQ_DATASET = "datascience_dev" if GOOGLE_CLOUD_PROJECT == "data-engineering-sandpit-70a1" else "dsci_pricing_model"
DAW_BQ_DATASET = "datascience_dev" if GOOGLE_CLOUD_PROJECT == "data-engineering-sandpit-70a1" else "dsci_daw"

bucket_lu = {
    "mpb-data-science-dev-oh-abc4": "data-science-bucket-dev-oh-sha3",
    "data-engineering-sandpit-70a1": "data-science-sanpit-bucket",
    "mpb-data-science-prod-cd30": "data-science-bucket-fz6v",
    "mpb-data-science-dev-ab-602d": "data-science-bucket-dev-ab-mq0o",
}

GCS_BUCKET = bucket_lu[GOOGLE_CLOUD_PROJECT]
