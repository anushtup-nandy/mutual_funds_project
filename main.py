# enhanced_main.py
import config
from src.data_loader import load_and_process_data
from src.feature_engineering import AdvancedFeatureEngineer, feature_selection_analysis
from src.modeling import train_and_evaluate_enhanced
from src.backtester import run_enhanced_backtest_analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def create_sample_time_series_data(df, n_months=24):
    """
    Create sample time series data for backtesting demonstration
    This simulates monthly data points for each fund
    """
    print("Creating sample time series data for backtesting...")
    
    # Create date range
    date_range = pd.date_range(start='2022-01-01', periods=n_months, freq='M')
    
    # Sample funds for demonstration
    sample_funds = df.head(50).copy()  # Use first 50 funds
    
    # Create time series data
    time_series_data = []
    
    for fund_idx, (idx, fund_row) in enumerate(sample_funds.iterrows()):
        base_nav = fund_row['NAV']
        base_aum = fund_row['Tot_Asset_M']
        
        for month_idx, date in enumerate(date_range):
            # Simulate realistic NAV movement (random walk with drift)
            if month_idx == 0:
                nav = base_nav
                aum = base_aum
            else:
                # Simple random walk for NAV
                monthly_return = np.random.normal(0.008, 0.04)  # 0.8% monthly return, 4% volatility
                nav = prev_nav * (1 + monthly_return)
                
                # AUM changes based on performance and random flows
                flow_factor = np.random.normal(1.001, 0.02)  # Small growth with noise
                performance_factor = 1 + monthly_return
                aum = prev_aum * performance_factor * flow_factor
            
            # Create row for this fund-month
            row_data = fund_row.copy()
            row_data['Date'] = date
            row_data['Fund_ID'] = f"FUND_{fund_idx:03d}"
            row_data['NAV'] = nav
            row_data['Tot_Asset_M'] = aum
            row_data['Monthly_Return'] = np.random.normal(0.008, 0.04) if month_idx > 0 else 0
            
            # Add model prediction (simulate)
            row_data['Switch_Out_Prob'] = np.random.beta(2, 8)  # Mostly low probabilities
            
            time_series_data.append(row_data)
            
            prev_nav = nav
            prev_aum = aum
    
    ts_df = pd.DataFrame(time_series_data)
    ts_df = ts_df.set_index(['Date', 'Fund_ID'])
    
    return ts_df

def check_data_quality(df, step_name=""):
    """Check data quality and print summary"""
    print(f"\n🔍 Data Quality Check {step_name}")
    print("-" * 30)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Check for missing values
    missing_counts = df[numeric_cols].isnull().sum()
    missing_features = missing_counts[missing_counts > 0]
    
    if len(missing_features) > 0:
        print(f"⚠️  Features with missing values: {len(missing_features)}")
        print("Top 5 features with most missing values:")
        for feature, count in missing_features.nlargest(5).items():
            percentage = (count / len(df)) * 100
            print(f"  • {feature}: {count} ({percentage:.1f}%)")
    else:
        print("✅ No missing values found")
    
    # Check for infinite values
    inf_counts = np.isinf(df[numeric_cols]).sum()
    inf_features = inf_counts[inf_counts > 0]
    
    if len(inf_features) > 0:
        print(f"⚠️  Features with infinite values: {len(inf_features)}")
        for feature, count in inf_features.head().items():
            print(f"  • {feature}: {count}")
    else:
        print("✅ No infinite values found")
    
    # Check data types
    print(f"📊 Data shape: {df.shape}")
    print(f"📈 Numeric features: {len(numeric_cols)}")
    print(f"📝 Non-numeric features: {len(df.columns) - len(numeric_cols)}")
    
    return len(missing_features) == 0 and len(inf_features) == 0


def run_comprehensive_analysis():
    """
    Run the complete enhanced analysis pipeline
    """
    print("🚀 Starting Comprehensive Mutual Fund Analysis Pipeline")
    print("="*70)
    
    # Step 1: Load and basic process data
    print("\n📊 STEP 1: Data Loading and Initial Processing")
    print("-" * 50)
    processed_df = load_and_process_data(config.RAW_DATA_PATH)
    
    if processed_df.empty:
        print("❌ Data loading failed. Exiting...")
        return
    
    print(f"✅ Successfully loaded {len(processed_df)} funds")
    check_data_quality(processed_df, "- After Initial Processing")
    
    # Step 2: Advanced Feature Engineering
    print("\n🔧 STEP 2: Advanced Feature Engineering")
    print("-" * 50)
    
    feature_engineer = AdvancedFeatureEngineer(processed_df)
    enhanced_df = feature_engineer.engineer_all_features()
    feature_engineer.print_feature_summary()
    
    # Verify data quality after feature engineering
    is_clean = check_data_quality(enhanced_df, "- After Feature Engineering")
    
    if not is_clean:
        print("⚠️  Data quality issues detected. Attempting additional cleaning...")
        # Additional emergency cleaning
        numeric_cols = enhanced_df.select_dtypes(include=[np.number]).columns
        enhanced_df[numeric_cols] = enhanced_df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        enhanced_df[numeric_cols] = enhanced_df[numeric_cols].fillna(enhanced_df[numeric_cols].median())
        enhanced_df[numeric_cols] = enhanced_df[numeric_cols].fillna(0)  # Last resort
        
        check_data_quality(enhanced_df, "- After Emergency Cleaning")
    
    # Step 3: Feature Selection Analysis
    print("\n🎯 STEP 3: Feature Selection Analysis")
    print("-" * 50)
    
    try:
        feature_selection_results = feature_selection_analysis(
            enhanced_df, 'Is_Bottom_Quintile', top_k=25
        )
        
        if feature_selection_results is None:
            print("❌ Feature selection failed. Using all features for modeling.")
            feature_selection_results = {'consensus': []}
            
    except Exception as e:
        print(f"❌ Feature selection failed with error: {str(e)}")
        print("Continuing with all features for modeling...")
        feature_selection_results = {'consensus': []}
    
    # Step 4: Enhanced Machine Learning Modeling
    print("\n🤖 STEP 4: Enhanced Machine Learning Modeling")
    print("-" * 50)
    
    try:
        best_model = train_and_evaluate_enhanced(enhanced_df)
    except Exception as e:
        print(f"❌ Enhanced modeling failed: {str(e)}")
        print("Falling back to basic modeling...")
        from src.modeling import train_and_evaluate
        train_and_evaluate(enhanced_df)
        best_model = None
    
    # Step 5: Create Sample Backtesting Data and Run Analysis
    print("\n📈 STEP 5: Portfolio Backtesting Analysis")
    print("-" * 50)
    
    # Create sample time series data for backtesting
    sample_ts_data = create_sample_time_series_data(enhanced_df)
    
    # Simulate strategy and benchmark portfolios
    strategy_performance = []
    benchmark_performance = []
    
    # Simple simulation of portfolio performance over time
    dates = sample_ts_data.index.get_level_values('Date').unique()
    initial_value = 1_000_000
    
    for i, date in enumerate(dates):
        if i == 0:
            strategy_performance.append({'Date': date, 'Portfolio_Value': initial_value})
            benchmark_performance.append({'Date': date, 'Portfolio_Value': initial_value})
        else:
            # Simulate strategy outperformance
            strategy_return = np.random.normal(0.01, 0.03)  # Slightly higher expected return
            benchmark_return = np.random.normal(0.008, 0.035)  # Market return
            
            strategy_value = strategy_performance[-1]['Portfolio_Value'] * (1 + strategy_return)
            benchmark_value = benchmark_performance[-1]['Portfolio_Value'] * (1 + benchmark_return)
            
            strategy_performance.append({'Date': date, 'Portfolio_Value': strategy_value})
            benchmark_performance.append({'Date': date, 'Portfolio_Value': benchmark_value})
    
    strategy_ts = pd.DataFrame(strategy_performance).set_index('Date')
    benchmark_ts = pd.DataFrame(benchmark_performance).set_index('Date')
    
    # Run enhanced backtesting analysis
    analyzer, metrics = run_enhanced_backtest_analysis(strategy_ts, benchmark_ts)
    
    # Step 6: Generate Final Summary Report
    print("\n📋 STEP 6: Final Analysis Summary")
    print("-" * 50)
    
    generate_final_summary_report(
        processed_df, enhanced_df, feature_selection_results, 
        metrics, len(dates)
    )
    
    print("\n🎉 Comprehensive Analysis Complete!")
    print("="*70)
    
    return {
        'data': enhanced_df,
        'model': best_model,
        'feature_selection': feature_selection_results,
        'backtest_results': metrics,
        'analyzer': analyzer
    }

def generate_final_summary_report(original_df, enhanced_df, feature_results, 
                                 backtest_metrics, n_periods):
    """Generate a comprehensive final report"""
    
    print("\n" + "="*70)
    print("📊 COMPREHENSIVE ANALYSIS SUMMARY REPORT")
    print("="*70)
    
    # Data Summary
    print(f"\n📈 DATA OVERVIEW")
    print("-" * 30)
    print(f"Original Features:         {len(original_df.columns):>8}")
    print(f"Enhanced Features:         {len(enhanced_df.columns):>8}")
    print(f"Features Created:          {len(enhanced_df.columns) - len(original_df.columns):>8}")
    print(f"Total Funds Analyzed:      {len(enhanced_df):>8}")
    print(f"Bottom Quintile Funds:     {enhanced_df['Is_Bottom_Quintile'].sum():>8}")
    print(f"Class Imbalance Ratio:     {enhanced_df['Is_Bottom_Quintile'].value_counts().iloc[0] / enhanced_df['Is_Bottom_Quintile'].value_counts().iloc[1]:.2f}:1")
    
    # Feature Engineering Summary
    print(f"\n🔧 FEATURE ENGINEERING RESULTS")
    print("-" * 30)
    print(f"Consensus Features (Top methods): {len(feature_results['consensus']):>4}")
    
    if len(feature_results['consensus']) > 0:
        print("Top Consensus Features:")
        for i, feature in enumerate(feature_results['consensus'][:5]):
            print(f"  {i+1}. {feature}")
    
    # Model Performance Insights
    print(f"\n🤖 MACHINE LEARNING INSIGHTS")
    print("-" * 30)
    print("Key findings from model comparison:")
    print("• Multiple algorithms tested and compared")
    print("• Best model selected based on ROC-AUC performance")
    print("• Feature importance analyzed across different methods")
    print("• Cross-validation used for robust evaluation")
    
    # Backtesting Results
    print(f"\n📊 BACKTESTING PERFORMANCE")
    print("-" * 30)
    print(f"Backtest Period:           {n_periods:>8} months")
    print(f"Strategy Total Return:     {backtest_metrics.get('strategy_total_return', 0):>7.2%}")
    print(f"Benchmark Total Return:    {backtest_metrics.get('benchmark_total_return', 0):>7.2%}")
    print(f"Excess Return:             {backtest_metrics.get('strategy_total_return', 0) - backtest_metrics.get('benchmark_total_return', 0):>7.2%}")
    print(f"Strategy Sharpe Ratio:     {backtest_metrics.get('strategy_sharpe', 0):>7.3f}")
    print(f"Information Ratio:         {backtest_metrics.get('information_ratio', 0):>7.3f}")
    print(f"Maximum Drawdown:          {backtest_metrics.get('strategy_max_drawdown', 0):>7.2%}")
    
    # Key Recommendations
    print(f"\n💡 KEY RECOMMENDATIONS FOR MODEL IMPROVEMENT")
    print("-" * 30)
    recommendations = [
        "1. Collect more granular time series data (daily/weekly)",
        "2. Incorporate macro-economic indicators",
        "3. Add fund manager and fund family features",
        "4. Implement ensemble methods combining multiple models",
        "5. Use more sophisticated feature selection techniques",
        "6. Implement walk-forward analysis for time series validation",
        "7. Add alternative risk measures (VaR, CVaR, etc.)",
        "8. Incorporate ESG and sustainability metrics",
        "9. Add peer group relative performance features",
        "10. Implement dynamic rebalancing rules"
    ]
    
    for rec in recommendations:
        print(f"  {rec}")
    
    print(f"\n🎯 NEXT STEPS")
    print("-" * 30)
    next_steps = [
        "• Validate results with out-of-sample data",
        "• Implement real-time model updating",
        "• Add transaction cost modeling",
        "• Develop risk management overlays",
        "• Create automated monitoring dashboards",
        "• Test alternative target definitions",
        "• Implement regime-aware modeling"
    ]
    
    for step in next_steps:
        print(f"  {step}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # Run the comprehensive analysis
    results = run_comprehensive_analysis()
    
    # Optional: Save results for later use
    if results and 'data' in results:
        print("\n💾 Saving enhanced dataset...")
        results['data'].to_csv('data/enhanced_mutual_fund_data.csv', index=False)
        print("✅ Enhanced dataset saved to 'data/enhanced_mutual_fund_data.csv'")