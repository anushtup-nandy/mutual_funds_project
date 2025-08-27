# config.py
from pathlib import Path

# --- File Paths ---
# Create a 'data' directory in your project folder and place the CSV inside it.
DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DATA_PATH = "data/CAPSTONE MUTUAL FUND.csv"

# --- Model & Data Parameters ---
TARGET_COLUMN = 'Sharpe_3Y'
PERFORMANCE_QUINTILE = 0.20  # We are targeting the bottom 20%
MIN_HISTORY_DAYS = 3 * 365 # Minimum history required for a fund to be included (for Sharpe 3Y)

# --- Machine Learning Parameters ---
TEST_SET_SIZE = 0.20
RANDOM_STATE = 42