# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Code Formatting and Linting
- `make cleanup` - Run ruff linting and formatting (excludes experiments/_legacy)
- `ruff check --fix .` - Run linting with auto-fixes
- `ruff format .` - Format code

### Dependencies
- This project uses `uv` for dependency management with Python 3.12+
- Install dependencies: `uv sync`
- Add new dependencies: `uv add <package>`
- Add development dependencies: `uv add --group dev <package>`

## Project Architecture

### Core Structure
This is a customer lifetime value (LTV) analysis project focused on propensity to lapse modeling in an infrequent purchase context.

**Key Directories:**
- `main.ipynb` - Primary analysis notebook documenting the overall approach
- `src/` - Core utilities, models, and configuration
- `experiments/` - Individual experimental notebooks for different modeling approaches
- `data/` - Dataset storage (parquet files)

### Source Code Organization (`src/`)

**Core Modules:**
- `config.py` - Project configuration, dataset paths, BigQuery queries, and constants
- `model.py` - Model implementations including BTYD models (BGNBD, Pareto/NBD) and survival analysis models
- `data.py` - Data processing utilities and BigQuery integration
- `eval.py` - Model evaluation metrics and utilities
- `utils/bq.py` - BigQuery utility classes
- `utils/logger.py` - Logging configuration

### Data Pipeline
The project uses BigQuery as the primary data source:
- Transaction data from `mpb-data-science-dev-ab-602d.dsci_daw.STV`
- Customer data from complex joins across multiple platform tables
- Data is cached locally as parquet files in `data/` directory

### Modeling Approaches
Three main approaches are implemented:
1. **BTYD Models** (Buy Till You Die) - BGNBD and Pareto/NBD models from `lifetimes` library
2. **Survival Analysis** - Cox Proportional Hazards and Random Survival Forest using `scikit-survival`
3. **Empirical Approaches** - Simple recency-based thresholds

### Key Dependencies
- `lifetimes` - BTYD modeling
- `lifelines` - Survival analysis
- `scikit-survival` - Advanced survival models
- `google-cloud-bigquery` - Data access
- `pandas`, `numpy` - Data manipulation
- `matplotlib`, `seaborn` - Visualization

### Configuration Constants
Key thresholds and labels defined in `config.py`:
- `CUTOFF_DAYS = 90` - Time threshold for analysis
- Customer states: `ACTIVE`, `LAPSING`, `LOST`
- Probability thresholds: alive_min=0.6, lapsed_max=0.3

### Working with the Codebase
- Start with `main.ipynb` for the overall analysis flow
- Individual experiments are in `experiments/` directory
- Models follow a consistent interface defined in `BaseModel` class
- Data loading and feature engineering utilities are centralized in `data.py`

### Model Evaluation
The evaluation framework in `eval.py` provides standardized metrics:

**P_alive (Binary Classification) Metrics:**
- `log_loss` - Cross-entropy loss (lower is better)
- `brier_score` - Quadratic scoring rule (lower is better) 
- `auc` - Area under ROC curve (higher is better)
- `ece` - Expected Calibration Error (lower is better)
- `concordance_index` - C-index for survival analysis (higher is better)

**Future Transactions (Regression) Metrics:**
- `mae` - Mean Absolute Error (lower is better)
- `rmse` - Root Mean Square Error (lower is better)
- `mape` - Mean Absolute Percentage Error (lower is better)
- `poisson_deviance` - Poisson deviance (lower is better)

**Key Evaluation Functions:**
- `ModelEvaluator.compare_models()` - Compare all models with consistent metrics
- `plot_all_calibrations()` - Transaction calibration plots in nx2 grid
- `_plot_roc_curve()` - ROC curves with AUC, Brier, Log Loss, and ECE
- `create_train_test_split()` - Proper temporal splitting without data leakage

### Train/Test Split Guidelines
Use `create_train_test_split()` for proper evaluation:
- Temporal split prevents data leakage
- Includes all customers with ≥1 calibration transaction (no selection bias)
- Train set has no ground truth labels
- Test set includes `y_true_txns` and `y_true_alive` for evaluation