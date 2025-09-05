# Customer Lapse Propensity Analysis

Customer lifetime value analysis project focused on propensity to lapse modeling in an infrequent purchase context.

## Structure

- `main.ipynb` - Main analysis and documentation
- `experiments/` - Experimental notebooks for different modeling approaches  
- `src/` - Core utilities, models, and configuration
- `data/` - Dataset storage (parquet files)
- `scripts/` - Production scripts

## Usage (Standalone Scripts)

**Production model:**
```bash
python scripts/lapse_propensity.py
```

Dependencies in `scripts/lapse_propensity_requirements.txt`

**Model testing:**
```bash
python scripts/test_lapse_propensity.py
```

Outputs useful plots and diagnostics for the model

## Repo Dependencies

Install with `uv sync` (requires Python 3.12+)