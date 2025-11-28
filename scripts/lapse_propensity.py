from src.utils import get_bq_helper, setup_logging
from src.lapse_propensity.pipeline import pipe

log = setup_logging("lapse_propensity")


def main():
    bq = get_bq_helper()
    pipe(bq)


if __name__ == "__main__":
    main()
