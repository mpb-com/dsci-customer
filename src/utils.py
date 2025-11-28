from src.definitions import (
    SQL_FOLDER,
    BQ_LOGGING_FOLDER,
    GOOGLE_CLOUD_PROJECT,
    BQ_DATASET,
    DAW_BQ_DATASET,
    LOGGING_FOLDER,
)
from dsci_utilities import BQHelper
from loguru import logger
import sys
import os
from datetime import datetime


def get_bq_helper(validate=False) -> BQHelper:
    """Set up BQHelper instance."""
    return BQHelper(
        billing_project_id=GOOGLE_CLOUD_PROJECT,
        write_project_id=GOOGLE_CLOUD_PROJECT,
        read_project_id=GOOGLE_CLOUD_PROJECT,
        write_dataset=BQ_DATASET,
        read_dataset=BQ_DATASET,
        daw_dataset=DAW_BQ_DATASET,
        sql_folder=SQL_FOLDER,
        logging_folder=BQ_LOGGING_FOLDER,
        validate=validate,
    )


def setup_logging(script_name: str):
    """
    Configure Loguru for a script with stdout and dated file output.

    Usage:
        from loguru import logger
        from src.utils import setup_script_logging

        setup_script_logging("base_tables")
        logger.info("This goes to both stdout and base_tables_20241110.log")
    """
    # Ensure logs directory exists
    os.makedirs(LOGGING_FOLDER, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Add stdout handler
    logger.add(sys.stdout, level="INFO")

    # Add dated file handler
    today = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(LOGGING_FOLDER, f"{script_name}_{today}.log")
    logger.add(log_file, level="DEBUG", rotation="1 day", retention="30 days")
