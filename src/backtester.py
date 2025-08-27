# src/enhanced_backtesting.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class EnhancedPortfolioAnalyzer:
    """Enhanced portfolio backtesting and analysis with comprehensive visualizations"""
    
    def __init__(self, strategy_ts, benchmark_ts):
        self.strategy_ts = strategy_ts
        self.benchmark_ts = benchmark_ts
        self.strategy_returns = self.calculate_returns(strategy_ts)
        self.benchmark_returns = self.calculate_returns(benchmark_ts)
        
    def calculate_returns(self, portfolio_ts):
        """Calculate returns from portfolio value time series"""
        return portfolio_ts['Portfolio_Value'].pct_change().dropna()
    
    def calculate_metrics(self):
        """Calculate comprehensive performance metrics"""
        metrics = {}
        
        # Basic return metrics
        metrics['strategy_total_return'] = (self.strategy_ts['Portfolio_Value'].iloc[-1] / 
                                          self.strategy_ts['Portfolio_Value'].iloc[0]) - 1
        metrics['benchmark_total_return'] = (self.benchmark_ts['Portfolio_Value'].iloc[-1] / 
                                           self.benchmark_ts['Portfolio_Value'].iloc[0]) - 1
        
        # Annualized returns
        years = len(self.strategy_returns) / 12
        metrics['strategy_annualized_return'] = (1 + metrics['strategy_total_return']) ** (1/years) - 1
        metrics['benchmark_annualized_return'] = (1 + metrics['benchmark_total_return']) ** (1/years) - 1
        
        # Risk metrics
        metrics['strategy_volatility'] = self.strategy_returns.std() * np.sqrt(12)
        metrics['benchmark_volatility'] = self.benchmark_returns.std() * np.sqrt(12)
        
        # Sharpe ratio
        risk_free_rate = 0.04
        metrics['strategy_sharpe'] = ((metrics['strategy_annualized_return'] - risk_free_rate) / 
                                     metrics['strategy_volatility'])
        metrics['benchmark_sharpe'] = ((metrics['benchmark_annualized_return'] - risk_free_rate) / 
                                      metrics['benchmark_volatility'])
        
        # Maximum Drawdown
        metrics['strategy_max_drawdown'] = self.calculate_max_drawdown(self.strategy_ts)
        metrics['benchmark_max_drawdown'] = self.calculate_max_drawdown(self.benchmark_ts)
        
        # Beta and Alpha
        beta, alpha = self.calculate_beta_alpha()
        metrics['beta'] = beta
        metrics['alpha'] = alpha
        
        # Win rate
        metrics['win_rate'] = (self.strategy_returns > 0).mean()
        
        # Information ratio
        excess_returns = self.strategy_returns - self.benchmark_returns.reindex(self.strategy_returns.index)
        metrics['information_ratio'] = excess_returns.mean() / excess_returns.std() * np.sqrt(12)
        
        # Sortino ratio
        downside_returns = self.strategy_returns[self.strategy_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(12)
        metrics['sortino_ratio'] = (metrics['strategy_annualized_return'] - risk_free_rate) / downside_deviation
        
        # Calmar ratio
        metrics['calmar_ratio'] = metrics['strategy_annualized_return'] / abs(metrics['strategy_max_drawdown'])
        
        return metrics
    
    def calculate_max_drawdown(self, portfolio_ts):
        """Calculate maximum drawdown"""
        roll_max = portfolio_ts['Portfolio_Value'].cummax()
        drawdown = portfolio_ts['Portfolio_Value'] / roll_max - 1.0
        return drawdown.min()
    
    def calculate_beta_alpha(self):
        """Calculate beta and alpha using linear regression"""
        # Align returns
        aligned_strategy = self.strategy_returns.dropna()
        aligned_benchmark = self.benchmark_returns.reindex(aligned_strategy.index).dropna()
        
        # Remove any remaining NaN pairs
        valid_pairs = pd.concat([aligned_strategy, aligned_benchmark], axis=1).dropna()
        
        if len(valid_pairs) < 2:
            return np.nan, np.nan
            
        strategy_clean = valid_pairs.iloc[:, 0]
        benchmark_clean = valid_pairs.iloc[:, 1]
        
        beta, alpha, r_value, p_value, std_err = stats.linregress(benchmark_clean, strategy_clean)
        return beta, alpha * 12  # Annualize alpha
    
    def create_comprehensive_plots(self):
        """Create comprehensive visualization suite"""
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        
        # Create the main dashboard
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. Cumulative Performance
        ax1 = fig.add_subplot(gs[0, :])
        
        # Normalize to start at 100
        strategy_norm = (self.strategy_ts['Portfolio_Value'] / self.strategy_ts['Portfolio_Value'].iloc[0]) * 100
        benchmark_norm = (self.benchmark_ts['Portfolio_Value'] / self.benchmark_ts['Portfolio_Value'].iloc[0]) * 100
        
        ax1.plot(strategy_norm.index, strategy_norm.values, label='Strategy', linewidth=2, color='navy')
        ax1.plot(benchmark_norm.index, benchmark_norm.values, label='Benchmark', linewidth=2, color='red', alpha=0.7)
        ax1.set_title('Cumulative Performance Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Value (Base 100)')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 2. Rolling Performance Metrics
        ax2 = fig.add_subplot(gs[1, 0])
        rolling_window = 12  # 12-month rolling
        
        rolling_strategy = self.strategy_returns.rolling(rolling_window).mean() * 12
        rolling_benchmark = self.benchmark_returns.rolling(rolling_window).mean() * 12
        
        ax2.plot(rolling_strategy.index, rolling_strategy.values, label='Strategy', color='navy')
        ax2.plot(rolling_benchmark.index, rolling_benchmark.values, label='Benchmark', color='red', alpha=0.7)
        ax2.set_title('12-Month Rolling Returns', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Annualized Return')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Drawdown Analysis
        ax3 = fig.add_subplot(gs[1, 1])
        
        strategy_drawdown = self.calculate_rolling_drawdown(self.strategy_ts)
        benchmark_drawdown = self.calculate_rolling_drawdown(self.benchmark_ts)
        
        ax3.fill_between(strategy_drawdown.index, strategy_drawdown.values, 0, 
                        alpha=0.3, color='navy', label='Strategy')
        ax3.fill_between(benchmark_drawdown.index, benchmark_drawdown.values, 0, 
                        alpha=0.3, color='red', label='Benchmark')
        ax3.set_title('Drawdown Analysis', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Drawdown %')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Return Distribution
        ax4 = fig.add_subplot(gs[1, 2])
        
        ax4.hist(self.strategy_returns, bins=30, alpha=0.6, label='Strategy', color='navy', density=True)
        ax4.hist(self.benchmark_returns.reindex(self.strategy_returns.index), 
                bins=30, alpha=0.6, label='Benchmark', color='red', density=True)
        ax4.set_title('Monthly Return Distribution', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Monthly Return')
        ax4.set_ylabel('Density')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Risk-Return Scatter
        ax5 = fig.add_subplot(gs[2, 0])
        
        # Calculate rolling risk-return metrics
        rolling_ret_strategy = self.strategy_returns.rolling(12).mean() * 12
        rolling_vol_strategy = self.strategy_returns.rolling(12).std() * np.sqrt(12)
        rolling_ret_benchmark = self.benchmark_returns.rolling(12).mean() * 12
        rolling_vol_benchmark = self.benchmark_returns.rolling(12).std() * np.sqrt(12)
        
        ax5.scatter(rolling_vol_strategy, rolling_ret_strategy, alpha=0.6, 
                   color='navy', label='Strategy', s=20)
        ax5.scatter(rolling_vol_benchmark.reindex(rolling_vol_strategy.index), 
                   rolling_ret_benchmark.reindex(rolling_ret_strategy.index), 
                   alpha=0.6, color='red', label='Benchmark', s=20)
        ax5.set_title('Risk-Return Profile (12M Rolling)', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Volatility (Annualized)')
        ax5.set_ylabel('Return (Annualized)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Beta Analysis
        ax6 = fig.add_subplot(gs[2, 1])
        
        # Calculate rolling beta
        rolling_beta = self.calculate_rolling_beta(window=12)
        ax6.plot(rolling_beta.index, rolling_beta.values, color='green', linewidth=2)
        ax6.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Beta = 1')
        ax6.set_title('12-Month Rolling Beta', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Beta')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Performance Metrics Table
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        
        metrics = self.calculate_metrics()
        
        # Create metrics table
        table_data = [
            ['Metric', 'Strategy', 'Benchmark'],
            ['Total Return', f"{metrics['strategy_total_return']:.2%}", f"{metrics['benchmark_total_return']:.2%}"],
            ['Ann. Return', f"{metrics['strategy_annualized_return']:.2%}", f"{metrics['benchmark_annualized_return']:.2%}"],
            ['Volatility', f"{metrics['strategy_volatility']:.2%}", f"{metrics['benchmark_volatility']:.2%}"],
            ['Sharpe Ratio', f"{metrics['strategy_sharpe']:.3f}", f"{metrics['benchmark_sharpe']:.3f}"],
            ['Max Drawdown', f"{metrics['strategy_max_drawdown']:.2%}", f"{metrics['benchmark_max_drawdown']:.2%}"],
            ['Information Ratio', f"{metrics['information_ratio']:.3f}", 'N/A'],
            ['Win Rate', f"{metrics['win_rate']:.2%}", 'N/A'],
            ['Beta', f"{metrics['beta']:.3f}", '1.000'],
            ['Alpha (Ann.)', f"{metrics['alpha']:.2%}", '0.00%']
        ]
        
        table = ax7.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Style the table
        for i in range(len(table_data)):
            for j in range(len(table_data[0])):
                cell = table[(i, j)]
                if i == 0:  # Header row
                    cell.set_facecolor('#4472C4')
                    cell.set_text_props(weight='bold', color='white')
                elif j == 1:  # Strategy column
                    cell.set_facecolor('#E8F1FF')
                elif j == 2:  # Benchmark column
                    cell.set_facecolor('#FFE8E8')
        
        ax7.set_title('Performance Summary', fontsize=12, fontweight='bold', pad=20)
        
        # 8. Monthly Returns Heatmap
        ax8 = fig.add_subplot(gs[3, :2])
        
        # Create monthly returns pivot table
        strategy_monthly = self.strategy_returns.to_frame('Returns')
        strategy_monthly['Year'] = strategy_monthly.index.year
        strategy_monthly['Month'] = strategy_monthly.index.month
        
        heatmap_data = strategy_monthly.pivot_table(
            values='Returns', index='Year', columns='Month', aggfunc='sum'
        )
        
        # Create heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='.2%', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Monthly Return'}, ax=ax8)
        ax8.set_title('Monthly Returns Heatmap - Strategy', fontsize=12, fontweight='bold')
        ax8.set_xlabel('Month')
        ax8.set_ylabel('Year')
        
        # 9. Statistical Tests Results
        ax9 = fig.add_subplot(gs[3, 2])
        ax9.axis('off')
        
        # Perform statistical tests
        aligned_returns = pd.concat([
            self.strategy_returns, 
            self.benchmark_returns.reindex(self.strategy_returns.index)
        ], axis=1).dropna()
        
        if len(aligned_returns) > 1:
            t_stat, t_pvalue = stats.ttest_rel(aligned_returns.iloc[:, 0], aligned_returns.iloc[:, 1])
            
            # Jarque-Bera normality test
            jb_stat, jb_pvalue = stats.jarque_bera(self.strategy_returns.dropna())
            
            # Sharpe ratio difference test
            sharpe_diff = metrics['strategy_sharpe'] - metrics['benchmark_sharpe']
            
            test_results = [
                ['Statistical Tests', 'Result'],
                ['T-Test p-value', f"{t_pvalue:.4f}"],
                ['T-Test significant?', 'Yes' if t_pvalue < 0.05 else 'No'],
                ['Normality (JB p-val)', f"{jb_pvalue:.4f}"],
                ['Sharpe Difference', f"{sharpe_diff:.3f}"],
                ['', ''],
                ['Risk Metrics', ''],
                ['Sortino Ratio', f"{metrics['sortino_ratio']:.3f}"],
                ['Calmar Ratio', f"{metrics['calmar_ratio']:.3f}"],
            ]
        else:
            test_results = [['Statistical Tests', 'Result'], ['Insufficient Data', 'N/A']]
        
        test_table = ax9.table(cellText=test_results, loc='center', cellLoc='center')
        test_table.auto_set_font_size(False)
        test_table.set_fontsize(10)
        test_table.scale(1, 1.2)
        
        # Style the test results table
        for i in range(len(test_results)):
            for j in range(len(test_results[0])):
                cell = test_table[(i, j)]
                if i == 0 or (i < len(test_results) and test_results[i][0] in ['Risk Metrics']):
                    cell.set_facecolor('#4472C4')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#F8F9FA')
        
        ax9.set_title('Statistical Analysis', fontsize=12, fontweight='bold', pad=20)
        
        plt.suptitle('Portfolio Performance Dashboard', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()
        
        return metrics
    
    def calculate_rolling_drawdown(self, portfolio_ts):
        """Calculate rolling drawdown"""
        roll_max = portfolio_ts['Portfolio_Value'].cummax()
        drawdown = (portfolio_ts['Portfolio_Value'] / roll_max - 1.0) * 100
        return drawdown
    
    def calculate_rolling_beta(self, window=12):
        """Calculate rolling beta"""
        aligned_strategy = self.strategy_returns.dropna()
        aligned_benchmark = self.benchmark_returns.reindex(aligned_strategy.index).dropna()
        
        rolling_beta = []
        dates = []
        
        for i in range(window, len(aligned_strategy)):
            strategy_window = aligned_strategy.iloc[i-window:i]
            benchmark_window = aligned_benchmark.iloc[i-window:i]
            
            if len(strategy_window) == window and len(benchmark_window) == window:
                beta, _, _, _, _ = stats.linregress(benchmark_window, strategy_window)
                rolling_beta.append(beta)
                dates.append(aligned_strategy.index[i])
        
        return pd.Series(rolling_beta, index=dates)
    
    def generate_detailed_report(self):
        """Generate a comprehensive text report"""
        metrics = self.calculate_metrics()
        
        print("=" * 60)
        print("COMPREHENSIVE PORTFOLIO PERFORMANCE REPORT")
        print("=" * 60)
        
        print("\n📈 RETURN ANALYSIS")
        print("-" * 30)
        print(f"Strategy Total Return:     {metrics['strategy_total_return']:>8.2%}")
        print(f"Benchmark Total Return:    {metrics['benchmark_total_return']:>8.2%}")
        print(f"Excess Return:             {metrics['strategy_total_return'] - metrics['benchmark_total_return']:>8.2%}")
        print(f"Strategy Ann. Return:      {metrics['strategy_annualized_return']:>8.2%}")
        print(f"Benchmark Ann. Return:     {metrics['benchmark_annualized_return']:>8.2%}")
        
        print(f"\n📊 RISK ANALYSIS")
        print("-" * 30)
        print(f"Strategy Volatility:       {metrics['strategy_volatility']:>8.2%}")
        print(f"Benchmark Volatility:      {metrics['benchmark_volatility']:>8.2%}")
        print(f"Strategy Max Drawdown:     {metrics['strategy_max_drawdown']:>8.2%}")
        print(f"Benchmark Max Drawdown:    {metrics['benchmark_max_drawdown']:>8.2%}")
        print(f"Beta:                      {metrics['beta']:>8.3f}")
        
        print(f"\n🎯 RISK-ADJUSTED RETURNS")
        print("-" * 30)
        print(f"Strategy Sharpe Ratio:     {metrics['strategy_sharpe']:>8.3f}")
        print(f"Benchmark Sharpe Ratio:    {metrics['benchmark_sharpe']:>8.3f}")
        print(f"Information Ratio:         {metrics['information_ratio']:>8.3f}")
        print(f"Sortino Ratio:             {metrics['sortino_ratio']:>8.3f}")
        print(f"Calmar Ratio:              {metrics['calmar_ratio']:>8.3f}")
        print(f"Alpha (Annualized):        {metrics['alpha']:>8.2%}")
        
        print(f"\n📈 ADDITIONAL METRICS")
        print("-" * 30)
        print(f"Win Rate:                  {metrics['win_rate']:>8.2%}")
        print(f"Best Month:                {self.strategy_returns.max():>8.2%}")
        print(f"Worst Month:               {self.strategy_returns.min():>8.2%}")
        print(f"Positive Months:           {(self.strategy_returns > 0).sum():>8d}")
        print(f"Negative Months:           {(self.strategy_returns < 0).sum():>8d}")
        
        # Performance vs benchmark assessment
        print(f"\n🏆 PERFORMANCE ASSESSMENT")
        print("-" * 30)
        
        outperformance_score = 0
        assessments = []
        
        if metrics['strategy_total_return'] > metrics['benchmark_total_return']:
            assessments.append("✅ Outperformed benchmark in total returns")
            outperformance_score += 1
        else:
            assessments.append("❌ Underperformed benchmark in total returns")
        
        if metrics['strategy_sharpe'] > metrics['benchmark_sharpe']:
            assessments.append("✅ Superior risk-adjusted returns (Sharpe)")
            outperformance_score += 1
        else:
            assessments.append("❌ Inferior risk-adjusted returns (Sharpe)")
        
        if metrics['strategy_max_drawdown'] > metrics['benchmark_max_drawdown']:
            assessments.append("❌ Higher maximum drawdown")
        else:
            assessments.append("✅ Lower maximum drawdown")
            outperformance_score += 1
        
        if metrics['strategy_volatility'] < metrics['benchmark_volatility']:
            assessments.append("✅ Lower volatility")
            outperformance_score += 1
        else:
            assessments.append("❌ Higher volatility")
        
        for assessment in assessments:
            print(assessment)
        
        print(f"\nOverall Score: {outperformance_score}/4")
        
        if outperformance_score >= 3:
            print("🎉 EXCELLENT: Strategy significantly outperformed benchmark")
        elif outperformance_score == 2:
            print("👍 GOOD: Strategy showed mixed but promising results")
        else:
            print("⚠️  NEEDS IMPROVEMENT: Strategy underperformed benchmark")
        
        print("=" * 60)
        
        return metrics


def run_enhanced_backtest_analysis(strategy_ts, benchmark_ts):
    """Main function to run enhanced backtesting analysis"""
    
    print("🚀 Starting Enhanced Portfolio Analysis...")
    
    # Initialize analyzer
    analyzer = EnhancedPortfolioAnalyzer(strategy_ts, benchmark_ts)
    
    # Generate comprehensive visualizations
    print("📊 Creating comprehensive visualizations...")
    metrics = analyzer.create_comprehensive_plots()
    
    # Generate detailed report
    print("📋 Generating detailed performance report...")
    analyzer.generate_detailed_report()
    
    return analyzer, metrics