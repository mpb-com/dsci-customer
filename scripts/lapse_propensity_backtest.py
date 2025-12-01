from src.lapse_propensity.backtest import backtest_pipeline
from src.utils import get_bq_helper, setup_logging

log = setup_logging("lapse_propensity_backtest")


def main():
    bq = get_bq_helper()
    backtest_pipeline(bq)
    log.info("Backtest pipeline completed")


if __name__ == "__main__":
    main()
