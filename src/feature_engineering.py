# src/feature_engineering.py
import pandas as pd
import numpy as np

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes remaining features and the target variable for the model.
    """
    # --- Basic Features from existing columns ---
    # Fund Age in months from History Length in days
    df['Fund_Age_Months'] = df['History_Len_Days'] / 30.44  # Average days in a month

    # Log of AUM
    df['Log_AUM'] = np.log(df['AUM'].replace(0, 1))

    # --- Calculate Net Flows (requires monthly data) ---
    gb = df.groupby(level='Fund_ID')
    # Return for the month is needed to adjust AUM change
    # Note: Your data provides Tot Ret 1M which is perfect for this.
    df['Monthly_Return'] = df['Trailing_1M_Return']
    
    # Previous month's AUM
    prev_aum = gb['AUM'].shift(1)
    
    # Net Flow = Current AUM - Previous AUM * (1 + Return)
    df['Net_Flows'] = df['AUM'] - prev_aum * (1 + df['Monthly_Return'])
    df['Net_Flows'] = df['Net_Flows'].fillna(0) # Fill NaNs for the first month of each fund

    # --- Categorical Features ---
    df['Age_Bucket'] = pd.cut(df['Fund_Age_Months'],
                              bins=[-np.inf, 36, 84, np.inf],
                              labels=['Early', 'Mid', 'Mature'])
    # Create dummy variables for style box
    df = pd.concat([df, pd.get_dummies(df['Style_Box'], prefix='Style', dummy_na=True)], axis=1)

    # --- Target Variable (y) ---
    # We still need to calculate next month's excess return vs. benchmark
    # Let's assume you have a benchmark CSV: ['Date', 'Benchmark_Return']
    # benchmark_df = pd.read_csv('path_to_benchmark.csv', parse_dates=['Date']).set_index('Date')
    # df = df.join(benchmark_df)
    # df['Excess_Return'] = df['Monthly_Return'] - df['Benchmark_Return']
    
    # For now, we'll create the target based on raw returns (can be updated later)
    df['Excess_Return'] = df['Monthly_Return']
    
    # IMPORTANT: Shift target back by 1 month to prevent look-ahead bias
    df['Target_Next_Month_Excess_Return'] = gb['Excess_Return'].shift(-1)
    
    # Define the "Switch-Out" label
    def quintile_label(x):
        return pd.qcut(x, q=5, labels=False, duplicates='drop') == 0
        
    df['Switch_Out_Label'] = df.groupby(level='Date')['Target_Next_Month_Excess_Return'].apply(quintile_label).astype(int)

    # Clean up NaNs created by shifts and calculations
    # Use a more specific list based on final predictors
    final_predictors = [
        'Fund_Age_Months', 'Trailing_3M_Return', 'Volatility_1Y', 'Sharpe_3Y',
        'Log_AUM', 'Net_Flows', 'Expense_Ratio'
    ] + [col for col in df.columns if 'Style_' in col]
    
    df = df.dropna(subset=final_predictors + ['Switch_Out_Label'])
    
    print("Features and target variable computed.")
    return df