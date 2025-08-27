import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def calculate_returns(portfolio_ts: pd.DataFrame) -> pd.Series:
    """Calculates monthly returns from a portfolio value time series."""
    return portfolio_ts['Portfolio_Value'].pct_change().dropna()

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate=0.04) -> float:
    """Calculates the annualized Sharpe Ratio."""
    excess_returns = returns - (risk_free_rate / 12)
    return (excess_returns.mean() * 12) / (excess_returns.std() * np.sqrt(12))

def calculate_max_drawdown(portfolio_ts: pd.DataFrame) -> float:
    """Calculates the maximum drawdown."""
    roll_max = portfolio_ts['Portfolio_Value'].cummax()
    daily_drawdown = portfolio_ts['Portfolio_Value'] / roll_max - 1.0
    return daily_drawdown.min()

def calculate_jensens_alpha(strategy_returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate=0.04):
    """Calculates Jensen's Alpha."""
    excess_strategy_returns = strategy_returns - (risk_free_rate / 12)
    excess_benchmark_returns = benchmark_returns - (risk_free_rate / 12)
    
    beta, alpha, _, _, _ = stats.linregress(excess_benchmark_returns, excess_strategy_returns)
    
    # Alpha is monthly, annualize it
    return alpha * 12

def bootstrap_sharpe_diff(strategy_returns, benchmark_returns, n_bootstrap=1000):
    """Performs a bootstrap test to find the p-value for the difference in Sharpe Ratios."""
    sharpe_diffs = []
    
    # Concatenate returns for sampling
    combined_returns = pd.concat([strategy_returns, benchmark_returns], axis=1)
    combined_returns.columns = ['strategy', 'benchmark']
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        sample = combined_returns.sample(n=len(combined_returns), replace=True)
        
        sharpe_strategy = calculate_sharpe_ratio(sample['strategy'])
        sharpe_benchmark = calculate_sharpe_ratio(sample['benchmark'])
        sharpe_diffs.append(sharpe_strategy - sharpe_benchmark)

    observed_diff = calculate_sharpe_ratio(strategy_returns) - calculate_sharpe_ratio(benchmark_returns)
    
    # Calculate p-value (two-tailed test)
    p_value = np.sum(np.array(sharpe_diffs) > observed_diff) / n_bootstrap
    return min(p_value, 1 - p_value) * 2

def generate_performance_report(strategy_ts, benchmark_ts):
    """Generates and prints a full performance summary."""
    strat_returns = calculate_returns(strategy_ts)
    bench_returns = calculate_returns(benchmark_ts)
    
    print("--- Performance Analysis Report ---")
    print(f"Strategy Sharpe Ratio: {calculate_sharpe_ratio(strat_returns):.4f}")
    print(f"Benchmark Sharpe Ratio: {calculate_sharpe_ratio(bench_returns):.4f}")
    print("-" * 35)
    print(f"Strategy Max Drawdown: {calculate_max_drawdown(strategy_ts):.2%}")
    print(f"Benchmark Max Drawdown: {calculate_max_drawdown(benchmark_ts):.2%}")
    print("-" * 35)
    print(f"Jensen's Alpha (Annualized): {calculate_jensens_alpha(strat_returns, bench_returns):.4f}")
    print("-" * 35)
    
    # Statistical Tests
    t_stat, p_val_ttest = stats.ttest_rel(strat_returns, bench_returns.reindex(strat_returns.index))
    print(f"Paired t-test on monthly returns (p-value): {p_val_ttest:.4f}")
    
    p_val_bootstrap = bootstrap_sharpe_diff(strat_returns, bench_returns)
    print(f"Bootstrap Sharpe Difference (p-value): {p_val_bootstrap:.4f}")
    print("-" * 35)