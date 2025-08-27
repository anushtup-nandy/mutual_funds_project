import pandas as pd
import numpy as np
import config

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names to be Python-friendly (e.g., 'Tot Asset (M)' -> 'Tot_Asset_M')."""
    cols = df.columns
    new_cols = []
    for col in cols:
        new_col = col.strip().replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
        new_cols.append(new_col)
    df.columns = new_cols
    return df

def load_and_process_data(filepath: str) -> pd.DataFrame:
    """
    Loads raw data and performs all necessary cleaning, filtering, feature
    engineering, and target variable creation.
    """
    print("Step 1: Loading and Initial Cleaning...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The data file was not found at {filepath}")
        print("Please ensure 'CAPSTONE MUTUAL FUND.csv' is inside the 'data' folder.")
        return pd.DataFrame()

    df = clean_column_names(df)

    # --- Data Type Conversion & Cleaning ---
    numeric_cols = [
        'Tot_Asset_M', 'Median_Mkt_Cap_M', 'NAV', 'Expense_Ratio', 'Tot_Ret_1M',
        'Tot_Ret_3M', 'Tot_Ret_6M', 'Tot_Ret_1Y', 'Tot_Ret_3Y', 'Mean_Ret_5Y_Ann',
        'Sharpe_3Y', 'Std_Dev_1Y-M', 'Avg_Price_Cash_Flow', 'Beta_3Y',
        'Avg_Dvd_Yield', 'History_Length', 'Ret_Vs_Idx_3Y', 'Alpha_3Y'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"Loaded {len(df)} initial records.")

    # --- Filtering and Universe Creation ---
    # Filter for Open-End Funds only, as per the research design.
    df = df[df['Fund_Type'] == 'Open-End Fund'].copy()
    # Filter for funds with sufficient history for our target metric.
    df = df[df['History_Length'] >= config.MIN_HISTORY_DAYS].copy()
    # Drop rows where our target or key predictors are missing. This is a critical quality step.
    df = df.dropna(subset=[config.TARGET_COLUMN, 'Tot_Asset_M', 'Expense_Ratio', 'Tot_Ret_1Y']).copy()
    print(f"Universe defined. {len(df)} funds remaining after filtering.")

    # --- Feature Engineering ---
    print("Step 2: Engineering Features...")
    df['Fund_Age_Months'] = df['History_Length'] / 30.44  # Approximate conversion from days to months
    df['Log_AUM'] = np.log(df['Tot_Asset_M'].replace(0, 1)) # Log transform to handle skewness in AUM

    # --- Handle Remaining Missing Values (Median Imputation) ---
    # For remaining numerical columns, median imputation is a robust strategy against outliers.
    cols_to_impute = df.select_dtypes(include=np.number).columns
    for col in cols_to_impute:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Imputed missing values in '{col}' with median: {median_val:.2f}")

    # --- Target Variable Creation ---
    print("Step 3: Creating Target Variable...")
    # Define the threshold for the bottom quintile based on the target column.
    bottom_quintile_threshold = df[config.TARGET_COLUMN].quantile(config.PERFORMANCE_QUINTILE)
    # Create the binary target label. 1 if in the bottom quintile, 0 otherwise.
    df['Is_Bottom_Quintile'] = (df[config.TARGET_COLUMN] <= bottom_quintile_threshold).astype(int)
    
    print(f"Target variable 'Is_Bottom_Quintile' created based on '{config.TARGET_COLUMN} <= {bottom_quintile_threshold:.2f}'.")
    print(f"Class distribution:\n{df['Is_Bottom_Quintile'].value_counts(normalize=True)}")

    return df