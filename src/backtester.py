import pandas as pd
import numpy as np
import config
from collections import defaultdict

class PortfolioBacktester:
    """
    Simulates a mutual fund switching strategy over a historical period.
    Tracks portfolio value, holdings, and applies friction costs.
    """
    def __init__(self, data: pd.DataFrame, initial_capital=1_000_000):
        self.data = data
        self.initial_capital = initial_capital
        self.dates = sorted(self.data.index.get_level_values('Date').unique())
        
    def run(self, strategy: str):
        """Runs the backtest for a given strategy ('ml' or 'simple_rule')."""
        self.cash = self.initial_capital
        self.holdings = defaultdict(float) # {Fund_ID: units}
        self.portfolio_history = []
        
        # --- Run Benchmark First (Buy-and-Hold Equal Weight) ---
        benchmark_history = self._run_benchmark()
        
        # --- Run Main Strategy ---
        for i, date in enumerate(self.dates):
            if i == 0: continue # Skip first day for return calculations
            
            # Get data for the current and previous month
            prev_date = self.dates[i-1]
            current_month_data = self.data.loc[date]
            
            # 1. Mark-to-market: Update portfolio value based on last month's holdings
            current_value = self.cash
            if self.holdings:
                held_funds = list(self.holdings.keys())
                navs = current_month_data.loc[current_month_data.index.isin(held_funds)]['NAV']
                current_value += (pd.Series(self.holdings) * navs).sum()
            
            self.portfolio_history.append({'Date': date, 'Portfolio_Value': current_value})

            # 2. Rebalancing Logic (executed at month-end, effective next period)
            funds_to_sell = self._get_exit_signals(current_month_data, strategy)
            
            # 3. Process Sells
            proceeds = self._process_sells(funds_to_sell, current_month_data)
            self.cash += proceeds
            
            # 4. Process Buys
            funds_to_buy = self._get_entry_signals(current_month_data, funds_to_sell)
            self._process_buys(funds_to_buy, current_month_data)

        strategy_ts = pd.DataFrame(self.portfolio_history).set_index('Date')
        benchmark_ts = pd.DataFrame(benchmark_history).set_index('Date')
        return strategy_ts, benchmark_ts
        
    def _get_exit_signals(self, month_data, strategy):
        if not self.holdings:
            return []
        
        held_funds = list(self.holdings.keys())
        held_data = month_data.loc[month_data.index.isin(held_funds)]
        
        if strategy == 'ml':
            # Exit if prediction probability is above the threshold
            return held_data[held_data['Switch_Out_Prob'] >= config.ML_PROB_THRESHOLD].index.tolist()
        elif strategy == 'simple_rule':
            # Exit if fund age crosses the mature threshold
            return held_data[held_data['Fund_Age_Months'] > config.AGE_THRESHOLD_MATURE].index.tolist()
        else:
            raise ValueError("Invalid strategy specified.")
            
    def _get_entry_signals(self, month_data, funds_to_sell):
        # Universe: Must have a valid prediction, not be on the sell list
        eligible_funds = month_data.dropna(subset=['Switch_Out_Prob'])
        eligible_funds = eligible_funds[~eligible_funds.index.isin(funds_to_sell)]
        
        # Filter for "Hold" prediction
        hold_funds = eligible_funds[eligible_funds['Switch_Out_Prob'] < config.ML_PROB_THRESHOLD]
        
        # Find youngest third of the "Hold" funds
        age_cutoff = hold_funds['Fund_Age_Months'].quantile(config.YOUNGEST_QUANTILE)
        buy_candidates = hold_funds[hold_funds['Fund_Age_Months'] <= age_cutoff]
        
        return buy_candidates.index.tolist()

    def _process_sells(self, funds_to_sell, month_data):
        total_proceeds = 0
        for fund_id in funds_to_sell:
            units = self.holdings[fund_id]
            nav = month_data.loc[fund_id]['NAV']
            value = units * nav
            
            # Simplified friction: Assume 1% exit load if within 12 months (needs purchase date tracking for accuracy)
            # Simplified tax: Assume all gains are STCG for simplicity
            # A rigorous implementation requires tracking tax lots (FIFO/LIFO).
            #cost_basis = ... # This requires tracking purchase price, a major enhancement
            # gain = value - cost_basis
            # tax = max(0, gain) * config.STCG_RATE
            # net_value = value - tax
            
            total_proceeds += value # Simplified version without tax/loads
            del self.holdings[fund_id]
        return total_proceeds
        
    def _process_buys(self, funds_to_buy, month_data):
        if not funds_to_buy or self.cash <= 0:
            return
            
        investment_per_fund = self.cash / len(funds_to_buy)
        for fund_id in funds_to_buy:
            nav = month_data.loc[fund_id]['NAV']
            units_to_buy = investment_per_fund / nav
            self.holdings[fund_id] += units_to_buy
            
        self.cash = 0 # All available cash is invested
        
    def _run_benchmark(self):
        """Simulates a buy-and-hold, equal-weight portfolio of all available funds."""
        initial_universe = self.data.loc[self.dates[0]].index
        capital_per_fund = self.initial_capital / len(initial_universe)
        
        initial_navs = self.data.loc[(self.dates[0], initial_universe), 'NAV']
        initial_units = capital_per_fund / initial_navs
        
        history = [{'Date': self.dates[0], 'Portfolio_Value': self.initial_capital}]
        
        for date in self.dates[1:]:
            current_navs = self.data.loc[(date, initial_universe), 'NAV'].reindex(initial_units.index)
            # Handle funds that may have delisted by filling with last known NAV (or 0)
            current_navs = current_navs.fillna(method='ffill').fillna(0)
            current_value = (initial_units * current_navs).sum()
            history.append({'Date': date, 'Portfolio_Value': current_value})
        
        return history