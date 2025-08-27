# src/advanced_feature_engineering.py
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for mutual fund performance prediction.
    Creates sophisticated features based on financial theory and market behavior.
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.feature_descriptions = {}
        
    def create_momentum_features(self):
        """Create momentum-based features"""
        print("Creating momentum features...")
        
        # Price momentum (different time horizons)
        self.df['Momentum_1M_3M'] = self.df['Tot_Ret_1M'] - self.df['Tot_Ret_3M']
        self.df['Momentum_3M_6M'] = self.df['Tot_Ret_3M'] - self.df['Tot_Ret_6M']
        self.df['Momentum_6M_1Y'] = self.df['Tot_Ret_6M'] - self.df['Tot_Ret_1Y']
        
        # Cross-sectional momentum (rank-based)
        self.df['Return_Rank_1M'] = self.df['Tot_Ret_1M'].rank(pct=True)
        self.df['Return_Rank_3M'] = self.df['Tot_Ret_3M'].rank(pct=True)
        self.df['Return_Rank_1Y'] = self.df['Tot_Ret_1Y'].rank(pct=True)
        
        # Momentum strength
        returns_cols = ['Tot_Ret_1M', 'Tot_Ret_3M', 'Tot_Ret_6M', 'Tot_Ret_1Y']
        self.df['Momentum_Consistency'] = self.df[returns_cols].apply(
            lambda x: (x > 0).sum() / len(x), axis=1
        )
        
        # Acceleration (second derivative of performance)
        self.df['Return_Acceleration'] = (
            self.df['Tot_Ret_1M'] - 2*self.df['Tot_Ret_3M'] + self.df['Tot_Ret_6M']
        )
        
        self.feature_descriptions.update({
            'Momentum_1M_3M': 'Short-term vs medium-term momentum differential',
            'Return_Rank_1M': 'Cross-sectional ranking of 1-month returns',
            'Momentum_Consistency': 'Proportion of positive returns across time horizons',
            'Return_Acceleration': 'Rate of change in momentum (acceleration)'
        })
    
    def create_risk_features(self):
        """Create risk-based features"""
        print("Creating risk features...")
        
        # Risk-adjusted returns
        self.df['Sharpe_1Y'] = self.df['Tot_Ret_1Y'] / self.df['Std_Dev_1Y-M']
        self.df['Risk_Adjusted_Return'] = self.df['Tot_Ret_1Y'] / (self.df['Std_Dev_1Y-M'] + 1e-8)
        
        # Downside risk measures
        # Note: This would be more accurate with daily/weekly data
        self.df['Downside_Risk_Proxy'] = np.where(
            self.df['Tot_Ret_1Y'] < 0, 
            self.df['Std_Dev_1Y-M'] * 1.5,  # Approximate downside deviation
            self.df['Std_Dev_1Y-M'] * 0.8
        )
        
        # Risk-adjusted size
        self.df['Risk_Adjusted_AUM'] = self.df['Log_AUM'] / (self.df['Std_Dev_1Y-M'] + 1e-8)
        
        # Beta stability (would need time series data for true calculation)
        self.df['Beta_Risk'] = np.abs(self.df['Beta_3Y'] - 1.0)  # Distance from market beta
        
        # Concentration risk (approximation)
        self.df['Concentration_Risk'] = 1 / (self.df['Median_Mkt_Cap_M'] / 1000 + 1)
        
        # Risk efficiency
        self.df['Risk_Efficiency'] = self.df['Alpha_3Y'] / (self.df['Std_Dev_1Y-M'] + 1e-8)
        
        self.feature_descriptions.update({
            'Sharpe_1Y': 'One-year Sharpe ratio approximation',
            'Risk_Adjusted_AUM': 'Fund size adjusted for volatility',
            'Beta_Risk': 'Absolute deviation from market beta',
            'Risk_Efficiency': 'Alpha generation per unit of risk'
        })
    
    def create_fundamental_features(self):
        """Create fundamental analysis features"""
        print("Creating fundamental features...")
        
        # Expense efficiency
        self.df['Expense_Efficiency'] = self.df['Tot_Ret_1Y'] / (self.df['Expense_Ratio'] + 0.01)
        self.df['Expense_Rank'] = self.df['Expense_Ratio'].rank(pct=True)
        
        # Size-based features
        self.df['Size_Quintile'] = pd.qcut(self.df['Tot_Asset_M'], q=5, labels=[1,2,3,4,5])
        self.df['Is_Large_Fund'] = (self.df['Tot_Asset_M'] > self.df['Tot_Asset_M'].quantile(0.8)).astype(int)
        self.df['Is_Small_Fund'] = (self.df['Tot_Asset_M'] < self.df['Tot_Asset_M'].quantile(0.2)).astype(int)
        
        # Age-based features
        self.df['Is_New_Fund'] = (self.df['Fund_Age_Months'] < 24).astype(int)
        self.df['Is_Mature_Fund'] = (self.df['Fund_Age_Months'] > 60).astype(int)
        
        # Dividend features
        self.df['Dividend_Rank'] = self.df['Avg_Dvd_Yield'].rank(pct=True)
        self.df['High_Dividend'] = (self.df['Avg_Dvd_Yield'] > self.df['Avg_Dvd_Yield'].quantile(0.7)).astype(int)
        
        # Market cap exposure
        self.df['Market_Cap_Rank'] = self.df['Median_Mkt_Cap_M'].rank(pct=True)
        self.df['Large_Cap_Focus'] = (self.df['Median_Mkt_Cap_M'] > self.df['Median_Mkt_Cap_M'].quantile(0.7)).astype(int)
        
        # Value metrics
        if 'Avg_Price_Cash_Flow' in self.df.columns:
            self.df['PCF_Rank'] = self.df['Avg_Price_Cash_Flow'].rank(pct=True)
            self.df['Value_Score'] = 1 - self.df['PCF_Rank']  # Lower P/CF = higher value score
        
        self.feature_descriptions.update({
            'Expense_Efficiency': 'Return generated per unit of expense',
            'Size_Quintile': 'Fund size quintile ranking',
            'Value_Score': 'Value investment score based on P/CF ratio',
            'Market_Cap_Rank': 'Ranking based on median market cap exposure'
        })
    
    def create_interaction_features(self):
        """Create interaction features between key variables"""
        print("Creating interaction features...")
        
        # Size-performance interactions
        self.df['Size_Return_Interaction'] = self.df['Log_AUM'] * self.df['Tot_Ret_1Y']
        self.df['Size_Risk_Interaction'] = self.df['Log_AUM'] * self.df['Std_Dev_1Y-M']
        
        # Age-performance interactions
        self.df['Age_Return_Interaction'] = self.df['Fund_Age_Months'] * self.df['Tot_Ret_1Y']
        self.df['Age_Risk_Interaction'] = self.df['Fund_Age_Months'] * self.df['Std_Dev_1Y-M']
        
        # Expense-performance interactions
        self.df['Expense_Alpha_Interaction'] = self.df['Expense_Ratio'] * self.df['Alpha_3Y']
        self.df['Expense_Beta_Interaction'] = self.df['Expense_Ratio'] * self.df['Beta_3Y']
        
        # Risk-return interactions
        self.df['Risk_Return_Product'] = self.df['Std_Dev_1Y-M'] * self.df['Tot_Ret_1Y']
        self.df['Beta_Alpha_Product'] = self.df['Beta_3Y'] * self.df['Alpha_3Y']
        
        self.feature_descriptions.update({
            'Size_Return_Interaction': 'Interaction between fund size and returns',
            'Age_Return_Interaction': 'Interaction between fund age and performance',
            'Risk_Return_Product': 'Product of risk and return measures'
        })
    
    def create_percentile_features(self):
        """Create percentile-based features for relative ranking"""
        print("Creating percentile features...")
        
        key_metrics = ['Tot_Ret_1Y', 'Sharpe_3Y', 'Alpha_3Y', 'Expense_Ratio', 
                      'Tot_Asset_M', 'Fund_Age_Months', 'Std_Dev_1Y-M']
        
        for metric in key_metrics:
            if metric in self.df.columns:
                self.df[f'{metric}_Percentile'] = self.df[metric].rank(pct=True) * 100
                
                # Create quintile dummies
                quintiles = pd.qcut(self.df[metric], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
                quintile_dummies = pd.get_dummies(quintiles, prefix=f'{metric}_Quintile')
                self.df = pd.concat([self.df, quintile_dummies], axis=1)
        
        # Create composite scores
        if all(col in self.df.columns for col in ['Tot_Ret_1Y_Percentile', 'Sharpe_3Y_Percentile']):
            self.df['Performance_Composite'] = (
                self.df['Tot_Ret_1Y_Percentile'] * 0.4 + 
                self.df['Sharpe_3Y_Percentile'] * 0.6
            )
        
        if all(col in self.df.columns for col in ['Alpha_3Y_Percentile', 'Expense_Ratio_Percentile']):
            self.df['Alpha_Expense_Composite'] = (
                self.df['Alpha_3Y_Percentile'] * 0.7 + 
                (100 - self.df['Expense_Ratio_Percentile']) * 0.3  # Lower expense is better
            )
    
    def create_technical_features(self):
        """Create technical analysis inspired features"""
        print("Creating technical features...")
        
        # Trend features
        returns_series = ['Tot_Ret_1M', 'Tot_Ret_3M', 'Tot_Ret_6M', 'Tot_Ret_1Y']
        
        # Trend strength
        for i, col in enumerate(returns_series):
            if col in self.df.columns:
                self.df[f'{col}_Trend_Strength'] = self.df[col].apply(
                    lambda x: 1 if x > 0 else -1 if x < 0 else 0
                )
        
        # Moving average crossovers (approximation)
        if all(col in self.df.columns for col in ['Tot_Ret_1M', 'Tot_Ret_3M', 'Tot_Ret_6M']):
            self.df['Short_Term_MA'] = (self.df['Tot_Ret_1M'] + self.df['Tot_Ret_3M']) / 2
            self.df['Long_Term_MA'] = (self.df['Tot_Ret_6M'] + self.df['Tot_Ret_1Y']) / 2
            self.df['MA_Crossover'] = (
                self.df['Short_Term_MA'] > self.df['Long_Term_MA']
            ).astype(int)
        
        # Volatility regime
        vol_median = self.df['Std_Dev_1Y-M'].median()
        self.df['High_Vol_Regime'] = (self.df['Std_Dev_1Y-M'] > vol_median).astype(int)
        self.df['Low_Vol_Regime'] = (self.df['Std_Dev_1Y-M'] < vol_median * 0.8).astype(int)
        
        # Performance consistency
        if all(col in self.df.columns for col in returns_series):
            self.df['Performance_Volatility'] = self.df[returns_series].std(axis=1)
            self.df['Performance_Range'] = (
                self.df[returns_series].max(axis=1) - self.df[returns_series].min(axis=1)
            )
    
    def create_statistical_features(self):
        """Create statistically-based features"""
        print("Creating statistical features...")
        
        # Z-scores for key metrics
        for col in ['Tot_Ret_1Y', 'Alpha_3Y', 'Sharpe_3Y', 'Expense_Ratio']:
            if col in self.df.columns:
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                self.df[f'{col}_ZScore'] = (self.df[col] - mean_val) / std_val
                
                # Outlier flags
                self.df[f'{col}_Outlier'] = (np.abs(self.df[f'{col}_ZScore']) > 2).astype(int)
        
        # Skewness and kurtosis approximations
        returns_cols = ['Tot_Ret_1M', 'Tot_Ret_3M', 'Tot_Ret_6M', 'Tot_Ret_1Y']
        available_returns = [col for col in returns_cols if col in self.df.columns]
        
        if len(available_returns) >= 3:
            # Approximate skewness using return distribution
            self.df['Return_Skewness_Proxy'] = self.df[available_returns].apply(
                lambda x: stats.skew(x) if len(x.dropna()) > 2 else 0, axis=1
            )
            
            # Approximate kurtosis
            self.df['Return_Kurtosis_Proxy'] = self.df[available_returns].apply(
                lambda x: stats.kurtosis(x) if len(x.dropna()) > 2 else 0, axis=1
            )
    
    def create_regime_features(self):
        """Create market regime and condition features"""
        print("Creating regime features...")
        
        # Performance regimes based on returns
        self.df['Bull_Market_Performer'] = (self.df['Tot_Ret_1Y'] > self.df['Tot_Ret_1Y'].quantile(0.7)).astype(int)
        self.df['Bear_Market_Performer'] = (self.df['Tot_Ret_1Y'] < self.df['Tot_Ret_1Y'].quantile(0.3)).astype(int)
        
        # High beta vs low beta regimes
        beta_median = self.df['Beta_3Y'].median()
        self.df['High_Beta_Fund'] = (self.df['Beta_3Y'] > beta_median).astype(int)
        self.df['Low_Beta_Fund'] = (self.df['Beta_3Y'] < beta_median * 0.8).astype(int)
        
        # Alpha generation patterns
        self.df['Positive_Alpha'] = (self.df['Alpha_3Y'] > 0).astype(int)
        self.df['Strong_Alpha'] = (self.df['Alpha_3Y'] > self.df['Alpha_3Y'].quantile(0.8)).astype(int)
        
        # Expense efficiency tiers
        self.df['Low_Cost_Fund'] = (self.df['Expense_Ratio'] < self.df['Expense_Ratio'].quantile(0.3)).astype(int)
        self.df['High_Cost_Fund'] = (self.df['Expense_Ratio'] > self.df['Expense_Ratio'].quantile(0.7)).astype(int)
    
    def engineer_all_features(self):
        """Run all feature engineering methods"""
        print("Starting comprehensive feature engineering...")
        
        # Core feature engineering
        self.create_momentum_features()
        self.create_risk_features()
        self.create_fundamental_features()
        self.create_interaction_features()
        self.create_percentile_features()
        self.create_technical_features()
        self.create_statistical_features()
        self.create_regime_features()
        
        # Comprehensive data cleaning
        print("Performing comprehensive data cleaning...")
        
        # Step 1: Handle infinite values
        print("Replacing infinite values...")
        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        
        # Step 2: Identify numeric columns
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        
        # Step 3: Handle columns with too many missing values
        missing_threshold = 0.7  # Remove columns with >70% missing
        for col in numeric_columns:
            missing_ratio = self.df[col].isnull().sum() / len(self.df)
            if missing_ratio > missing_threshold:
                print(f"Dropping column {col} due to {missing_ratio:.2%} missing values")
                self.df = self.df.drop(columns=[col])
        
        # Update numeric columns list
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        
        # Step 4: Advanced imputation for remaining missing values
        from sklearn.impute import SimpleImputer
        
        # Separate features by missing value patterns
        low_missing = []  # <5% missing
        medium_missing = []  # 5-30% missing
        high_missing = []  # 30-70% missing
        
        for col in numeric_columns:
            missing_ratio = self.df[col].isnull().sum() / len(self.df)
            if missing_ratio < 0.05:
                low_missing.append(col)
            elif missing_ratio < 0.30:
                medium_missing.append(col)
            else:
                high_missing.append(col)
        
        # Different imputation strategies
        if low_missing:
            # Median imputation for low missing
            for col in low_missing:
                if self.df[col].isnull().sum() > 0:
                    median_val = self.df[col].median()
                    if pd.isna(median_val):
                        median_val = 0
                    self.df[col].fillna(median_val, inplace=True)
        
        if medium_missing:
            # More sophisticated imputation for medium missing
            imputer = SimpleImputer(strategy='median')
            self.df[medium_missing] = imputer.fit_transform(self.df[medium_missing])
        
        if high_missing:
            # Conservative imputation for high missing
            for col in high_missing:
                # Fill with median or 0 if median is NaN
                median_val = self.df[col].median()
                if pd.isna(median_val):
                    fill_val = 0
                else:
                    fill_val = median_val
                self.df[col].fillna(fill_val, inplace=True)
        
        # Step 5: Handle any remaining NaNs
        remaining_nan_cols = self.df.select_dtypes(include=[np.number]).columns[self.df.select_dtypes(include=[np.number]).isnull().any()]
        
        if len(remaining_nan_cols) > 0:
            print(f"Handling remaining NaNs in {len(remaining_nan_cols)} columns...")
            for col in remaining_nan_cols:
                # Try mode for categorical-like numeric columns
                if self.df[col].nunique() < 10:
                    mode_val = self.df[col].mode()
                    if len(mode_val) > 0:
                        self.df[col].fillna(mode_val.iloc[0], inplace=True)
                    else:
                        self.df[col].fillna(0, inplace=True)
                else:
                    # Use 0 for continuous variables as last resort
                    self.df[col].fillna(0, inplace=True)
        
        # Step 6: Remove features with zero or near-zero variance
        print("Removing zero-variance features...")
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col != 'Is_Bottom_Quintile':  # Don't remove target
                var_val = self.df[col].var()
                if pd.isna(var_val) or var_val < 1e-8:
                    print(f"Removing zero-variance feature: {col}")
                    self.df = self.df.drop(columns=[col])
        
        # Step 7: Final verification
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        total_nan = self.df[numeric_columns].isnull().sum().sum()
        total_inf = np.isinf(self.df[numeric_columns].select_dtypes(include=[np.number])).sum().sum()
        
        print(f"Final data quality check:")
        print(f"  - Remaining NaN values: {total_nan}")
        print(f"  - Remaining infinite values: {total_inf}")
        print(f"  - Final feature count: {len(self.df.columns)}")
        
        if total_nan > 0:
            print("Warning: NaN values still present. Performing emergency cleanup...")
            self.df = self.df.fillna(0)
        
        if total_inf > 0:
            print("Warning: Infinite values still present. Performing emergency cleanup...")
            self.df = self.df.replace([np.inf, -np.inf], 0)
        
        print("Feature engineering and data cleaning complete!")
        return self.df
    
    def get_feature_importance_groups(self):
        """Return feature groups for organized analysis"""
        feature_groups = {
            'momentum': [col for col in self.df.columns if 'Momentum' in col or 
                        'Return_Rank' in col or 'Acceleration' in col],
            'risk': [col for col in self.df.columns if 'Risk' in col or 'Sharpe' in col or 
                    'Beta' in col or 'Std_Dev' in col or 'Vol' in col],
            'fundamental': [col for col in self.df.columns if 'Expense' in col or 'Size' in col or 
                           'Age' in col or 'Dividend' in col or 'Market_Cap' in col],
            'interaction': [col for col in self.df.columns if 'Interaction' in col or 
                           'Product' in col],
            'percentile': [col for col in self.df.columns if 'Percentile' in col or 
                          'Quintile' in col or 'Composite' in col],
            'technical': [col for col in self.df.columns if 'MA' in col or 'Crossover' in col or 
                         'Trend' in col or 'Regime' in col],
            'statistical': [col for col in self.df.columns if 'ZScore' in col or 'Outlier' in col or 
                           'Skewness' in col or 'Kurtosis' in col],
            'regime': [col for col in self.df.columns if 'Bull' in col or 'Bear' in col or 
                      'Alpha' in col or 'Cost_Fund' in col]
        }
        return feature_groups
    
    def print_feature_summary(self):
        """Print a summary of created features"""
        print("\n" + "="*60)
        print("FEATURE ENGINEERING SUMMARY")
        print("="*60)
        
        feature_groups = self.get_feature_importance_groups()
        
        for group_name, features in feature_groups.items():
            if features:
                print(f"\n📊 {group_name.upper()} Features ({len(features)}):")
                for feature in features[:5]:  # Show first 5 features
                    description = self.feature_descriptions.get(feature, "No description available")
                    print(f"  • {feature}: {description}")
                if len(features) > 5:
                    print(f"  ... and {len(features) - 5} more")
        
        print(f"\nTotal features created: {len(self.df.columns)}")
        print("="*60)


def create_time_series_features(df, date_col='Date', fund_col='Fund_ID'):
    """
    Create time series features for panel data (if you have time series data)
    This function assumes you have a multi-index or separate date column
    """
    if date_col not in df.columns:
        print(f"Warning: {date_col} not found in dataframe. Skipping time series features.")
        return df
    
    print("Creating time series features...")
    
    # Convert date column if it's not datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Sort by fund and date
    df = df.sort_values([fund_col, date_col])
    
    # Create lagged features
    lag_features = ['Tot_Ret_1M', 'Tot_Asset_M', 'NAV', 'Expense_Ratio']
    
    for feature in lag_features:
        if feature in df.columns:
            # 1-month lag
            df[f'{feature}_Lag1'] = df.groupby(fund_col)[feature].shift(1)
            # 3-month lag  
            df[f'{feature}_Lag3'] = df.groupby(fund_col)[feature].shift(3)
            
            # Month-over-month change
            df[f'{feature}_MoM_Change'] = df[feature] - df[f'{feature}_Lag1']
            df[f'{feature}_MoM_Change_Pct'] = (df[feature] / df[f'{feature}_Lag1'] - 1) * 100
    
    # Rolling statistics (3-month and 6-month windows)
    rolling_features = ['Tot_Ret_1M', 'Tot_Asset_M', 'NAV']
    
    for feature in rolling_features:
        if feature in df.columns:
            # 3-month rolling statistics
            df[f'{feature}_Roll3_Mean'] = df.groupby(fund_col)[feature].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
            df[f'{feature}_Roll3_Std'] = df.groupby(fund_col)[feature].rolling(3, min_periods=1).std().reset_index(0, drop=True)
            
            # 6-month rolling statistics
            df[f'{feature}_Roll6_Mean'] = df.groupby(fund_col)[feature].rolling(6, min_periods=1).mean().reset_index(0, drop=True)
            df[f'{feature}_Roll6_Std'] = df.groupby(fund_col)[feature].rolling(6, min_periods=1).std().reset_index(0, drop=True)
    
    # Trend features
    for feature in ['Tot_Ret_1M', 'Tot_Asset_M']:
        if feature in df.columns:
            # Linear trend over last 6 months
            def calculate_trend(series):
                if len(series) < 2:
                    return 0
                x = np.arange(len(series))
                slope, _, _, _, _ = stats.linregress(x, series)
                return slope
            
            df[f'{feature}_Trend6M'] = df.groupby(fund_col)[feature].rolling(6, min_periods=2).apply(calculate_trend).reset_index(0, drop=True)
    
    # Seasonal features
    df['Month'] = df[date_col].dt.month
    df['Quarter'] = df[date_col].dt.quarter
    df['Year'] = df[date_col].dt.year
    
    # Create month and quarter dummies
    month_dummies = pd.get_dummies(df['Month'], prefix='Month')
    quarter_dummies = pd.get_dummies(df['Quarter'], prefix='Quarter')
    
    df = pd.concat([df, month_dummies, quarter_dummies], axis=1)
    
    print(f"Time series features created. New shape: {df.shape}")
    return df


def feature_selection_analysis(df, target_col, top_k=20):
    """
    Perform feature selection analysis to identify most important features
    """
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    import matplotlib.pyplot as plt
    
    print("Performing feature selection analysis...")
    
    # Prepare features (exclude non-numeric and target)
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in feature_cols:
        feature_cols.remove(target_col)
    
    print(f"Initial feature count: {len(feature_cols)}")
    
    # Get initial feature matrix
    X_raw = df[feature_cols]
    y = df[target_col]
    
    # Check for and handle infinite values
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan)
    
    # Advanced imputation strategy
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(
        imputer.fit_transform(X_raw), 
        columns=feature_cols, 
        index=X_raw.index
    )
    
    # Remove features with too many missing values (>50% before imputation)
    missing_ratio = X_raw.isnull().sum() / len(X_raw)
    valid_features = missing_ratio[missing_ratio < 0.5].index.tolist()
    
    if len(valid_features) < len(feature_cols):
        print(f"Removed {len(feature_cols) - len(valid_features)} features with >50% missing values")
        X = X[valid_features]
        feature_cols = valid_features
    
    # Remove features with zero variance
    feature_variance = X.var()
    non_zero_var_features = feature_variance[feature_variance > 1e-8].index.tolist()
    
    if len(non_zero_var_features) < len(feature_cols):
        print(f"Removed {len(feature_cols) - len(non_zero_var_features)} features with zero/near-zero variance")
        X = X[non_zero_var_features]
        feature_cols = non_zero_var_features
    
    print(f"Final feature count for selection: {len(feature_cols)}")
    
    # Ensure no NaNs remain
    if X.isnull().sum().sum() > 0:
        print("Warning: NaN values still present after imputation. Applying additional cleaning...")
        X = X.fillna(X.median())
        # If still NaN (all values were NaN), fill with 0
        X = X.fillna(0)
    
    # Ensure no infinite values remain
    X = X.replace([np.inf, -np.inf], 0)
    
    # Final check
    if X.isnull().sum().sum() > 0 or np.isinf(X.values).sum() > 0:
        print("Error: Still have invalid values after cleaning")
        return None
    
    # Limit top_k to available features
    top_k = min(top_k, len(feature_cols))
    
    # 1. Univariate Feature Selection (F-test)
    selector_f = SelectKBest(score_func=f_classif, k=top_k)
    X_selected_f = selector_f.fit_transform(X, y)
    f_scores = pd.DataFrame({
        'Feature': [feature_cols[i] for i in selector_f.get_support(indices=True)],
        'F_Score': selector_f.scores_[selector_f.get_support(indices=True)]
    }).sort_values('F_Score', ascending=False)
    
    # 2. Mutual Information
    selector_mi = SelectKBest(score_func=mutual_info_classif, k=top_k)
    X_selected_mi = selector_mi.fit_transform(X, y)
    mi_scores = pd.DataFrame({
        'Feature': [feature_cols[i] for i in selector_mi.get_support(indices=True)],
        'MI_Score': selector_mi.scores_[selector_mi.get_support(indices=True)]
    }).sort_values('MI_Score', ascending=False)
    
    # 3. Random Forest Feature Importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_scores = pd.DataFrame({
        'Feature': feature_cols,
        'RF_Importance': rf.feature_importances_
    }).sort_values('RF_Importance', ascending=False).head(top_k)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # F-test scores
    axes[0].barh(range(len(f_scores)), f_scores['F_Score'])
    axes[0].set_yticks(range(len(f_scores)))
    axes[0].set_yticklabels(f_scores['Feature'], fontsize=8)
    axes[0].set_title('Top Features by F-Test Score')
    axes[0].set_xlabel('F-Score')
    
    # Mutual Information scores
    axes[1].barh(range(len(mi_scores)), mi_scores['MI_Score'])
    axes[1].set_yticks(range(len(mi_scores)))
    axes[1].set_yticklabels(mi_scores['Feature'], fontsize=8)
    axes[1].set_title('Top Features by Mutual Information')
    axes[1].set_xlabel('MI Score')
    
    # Random Forest importance
    axes[2].barh(range(len(rf_scores)), rf_scores['RF_Importance'])
    axes[2].set_yticks(range(len(rf_scores)))
    axes[2].set_yticklabels(rf_scores['Feature'], fontsize=8)
    axes[2].set_title('Top Features by Random Forest')
    axes[2].set_xlabel('Feature Importance')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("FEATURE SELECTION RESULTS")
    print("="*60)
    
    print(f"\nTop 10 Features by F-Test:")
    for i, row in f_scores.head(10).iterrows():
        print(f"  {row['Feature']}: {row['F_Score']:.2f}")
    
    print(f"\nTop 10 Features by Mutual Information:")
    for i, row in mi_scores.head(10).iterrows():
        print(f"  {row['Feature']}: {row['MI_Score']:.4f}")
    
    print(f"\nTop 10 Features by Random Forest:")
    for i, row in rf_scores.head(10).iterrows():
        print(f"  {row['Feature']}: {row['RF_Importance']:.4f}")
    
    # Find consensus features (appearing in multiple top lists)
    consensus_features = set(f_scores.head(10)['Feature']) & \
                        set(mi_scores.head(10)['Feature']) & \
                        set(rf_scores.head(10)['Feature'])
    
    print(f"\nConsensus Features (Top 10 in all methods): {len(consensus_features)}")
    for feature in consensus_features:
        print(f"  • {feature}")
    
    return {
        'f_test': f_scores,
        'mutual_info': mi_scores,
        'random_forest': rf_scores,
        'consensus': list(consensus_features)
    } 