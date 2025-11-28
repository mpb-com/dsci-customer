from loguru import logger as log
from .features import fetch_transactions, create_btyd_features
from .model import ParetoEmpiricalSingleTrainSplit
from .eval import log_dataframe_stats, log_config_constants
from .config import FINAL_COLUMNS, PROJECT_ID, DATABASE_NAME, TABLE_NAME


def pipe(bq):
    log.info("Starting lapse propensity model")

    # Fetch and process data
    log.info("Fetching transactions and calculating features")
    transactions = fetch_transactions(bq)
    features = create_btyd_features(transactions)

    # Train model
    log.info("Fitting model")
    model = ParetoEmpiricalSingleTrainSplit()
    model.fit(features)

    # Generate predictions
    log.info("Calculating probabilities and customer status")
    features["p_alive"] = model.p_alive(features).round(2)
    features["customer_status"] = model.customer_status(features)

    # Select final columns
    final_features = features[FINAL_COLUMNS]

    # Log results and config
    log_dataframe_stats(final_features, "Final results")
    log_config_constants()

    # Save results
    bq.write_to(final_features, f"{PROJECT_ID}.{DATABASE_NAME}.{TABLE_NAME}")
    log.info("Lapse propensity model completed and data saved to BigQuery")
