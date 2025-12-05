import argparse
from src.utils import get_bq_helper, setup_logging
from src.lapse_propensity.pipeline import pipe

log = setup_logging("lapse_propensity")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run lapse propensity production pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With calibration (default - uses business thresholds)
  python scripts/lapse_propensity.py

  # Without calibration (uses fixed 0.3/0.6 thresholds)
  python scripts/lapse_propensity.py --no-calibration
        """,
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Skip isotonic calibration and use raw predictions with fixed thresholds (0.3, 0.6)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    bq = get_bq_helper()
    use_calibration = not args.no_calibration

    mode = "WITH calibration" if use_calibration else "WITHOUT calibration (fixed thresholds)"
    log.info(f"Running production pipeline {mode}")

    pipe(bq, use_calibration=use_calibration)
    log.info("Production pipeline completed")


if __name__ == "__main__":
    main()
