# src/enhanced_modeling.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, f1_score, 
                           confusion_matrix, classification_report, roc_curve, precision_recall_curve)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import config

class EnhancedModelTrainer:
    def __init__(self, df):
        self.df = df
        self.models = {}
        self.results = {}
        
    def create_advanced_features(self):
        """Create more sophisticated features"""
        print("Creating advanced features...")
        
        # Rolling statistics (if you have time series data)
        if 'Date' in self.df.columns:
            self.df = self.df.sort_values('Date')
            
        # Momentum features
        self.df['Return_Momentum'] = (self.df['Tot_Ret_1M'] + self.df['Tot_Ret_3M']) / 2
        self.df['Volatility_Risk'] = self.df['Std_Dev_1Y-M'] / self.df['Tot_Ret_1Y'].abs()
        
        # Risk-adjusted returns
        self.df['Risk_Adjusted_Return'] = self.df['Tot_Ret_1Y'] / self.df['Std_Dev_1Y-M']
        self.df['Expense_Efficiency'] = self.df['Tot_Ret_1Y'] / self.df['Expense_Ratio']
        
        # Size quintiles
        self.df['Size_Quintile'] = pd.qcut(self.df['Tot_Asset_M'], q=5, labels=[1,2,3,4,5])
        
        # Performance consistency
        returns_cols = ['Tot_Ret_1M', 'Tot_Ret_3M', 'Tot_Ret_6M', 'Tot_Ret_1Y']
        self.df['Return_Consistency'] = self.df[returns_cols].std(axis=1)
        
        # Age buckets
        self.df['Age_Category'] = pd.cut(self.df['Fund_Age_Months'], 
                                        bins=[0, 12, 36, 60, np.inf], 
                                        labels=['New', 'Young', 'Mature', 'Old'])
        
        # One-hot encode categorical variables
        categorical_cols = ['Age_Category', 'Size_Quintile']
        for col in categorical_cols:
            if col in self.df.columns:
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                self.df = pd.concat([self.df, dummies], axis=1)
        
        print("Advanced features created successfully!")
        
    def prepare_data(self):
        """Enhanced data preparation with feature selection"""
        # Define comprehensive feature set
        base_features = [
            'Fund_Age_Months', 'Log_AUM', 'Expense_Ratio', 
            'Tot_Ret_1M', 'Tot_Ret_3M', 'Tot_Ret_6M', 'Tot_Ret_1Y',
            'Std_Dev_1Y-M', 'Beta_3Y', 'Avg_Dvd_Yield', 'Alpha_3Y',
            'Return_Momentum', 'Volatility_Risk', 'Risk_Adjusted_Return',
            'Expense_Efficiency', 'Return_Consistency'
        ]
        
        # Add dummy variables
        dummy_cols = [col for col in self.df.columns if 
                     any(prefix in col for prefix in ['Age_Category_', 'Size_Quintile_'])]
        
        all_features = base_features + dummy_cols
        available_features = [f for f in all_features if f in self.df.columns]
        
        print(f"Using {len(available_features)} features for modeling")
        
        X = self.df[available_features].fillna(self.df[available_features].median())
        y = self.df['Is_Bottom_Quintile']
        
        return X, y, available_features
    
    def train_multiple_models(self, X_train, X_test, y_train, y_test):
        """Train multiple models and compare performance"""
        print("\nTraining multiple models...")
        
        # Define models with hyperparameter grids
        model_configs = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 15, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'class_weight': ['balanced']
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=config.RANDOM_STATE),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            'Logistic Regression': {
                'model': Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', LogisticRegression(random_state=config.RANDOM_STATE, class_weight='balanced'))
                ]),
                'params': {
                    'classifier__C': [0.1, 1.0, 10.0],
                    'classifier__penalty': ['l1', 'l2'],
                    'classifier__solver': ['liblinear']
                }
            }
        }
        
        results = {}
        
        for name, config_dict in model_configs.items():
            print(f"Training {name}...")
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                config_dict['model'], 
                config_dict['params'],
                cv=5, 
                scoring='roc_auc',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # Predictions
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1]
            
            # Store results
            results[name] = {
                'model': best_model,
                'best_params': grid_search.best_params_,
                'y_pred': y_pred,
                'y_prob': y_prob,
                'roc_auc': roc_auc_score(y_test, y_prob),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }
            
            print(f"{name} ROC-AUC: {results[name]['roc_auc']:.4f}")
        
        return results
    
    def create_comprehensive_visualizations(self, X_train, X_test, y_train, y_test, 
                                          results, feature_names):
        """Create comprehensive visualization suite"""
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig_size = (15, 10)
        
        # 1. Model Comparison Dashboard
        fig, axes = plt.subplots(2, 2, figsize=fig_size)
        
        # ROC Curves
        ax1 = axes[0, 0]
        for name, result in results.items():
            fpr, tpr, _ = roc_curve(y_test, result['y_prob'])
            ax1.plot(fpr, tpr, label=f"{name} (AUC: {result['roc_auc']:.3f})")
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Precision-Recall Curves
        ax2 = axes[0, 1]
        for name, result in results.items():
            precision, recall, _ = precision_recall_curve(y_test, result['y_prob'])
            ax2.plot(recall, precision, label=f"{name}")
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Model Performance Metrics
        ax3 = axes[1, 0]
        metrics = ['roc_auc', 'precision', 'recall', 'f1']
        model_names = list(results.keys())
        x = np.arange(len(metrics))
        width = 0.25
        
        for i, name in enumerate(model_names):
            values = [results[name][metric] for metric in metrics]
            ax3.bar(x + i*width, values, width, label=name)
        
        ax3.set_xlabel('Metrics')
        ax3.set_ylabel('Score')
        ax3.set_title('Model Performance Comparison')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels(metrics)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Feature Importance (for tree-based models)
        ax4 = axes[1, 1]
        best_model_name = max(results.keys(), key=lambda k: results[k]['roc_auc'])
        best_model = results[best_model_name]['model']
        
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
        elif hasattr(best_model, 'named_steps') and hasattr(best_model.named_steps['classifier'], 'coef_'):
            importances = np.abs(best_model.named_steps['classifier'].coef_[0])
        else:
            importances = np.random.rand(len(feature_names))  # Fallback
            
        indices = np.argsort(importances)[::-1][:10]
        ax4.bar(range(len(indices)), importances[indices])
        ax4.set_title(f'Top 10 Feature Importances ({best_model_name})')
        ax4.set_xticks(range(len(indices)))
        ax4.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
        
        # 2. Data Distribution Analysis
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Target distribution
        axes[0, 0].pie(self.df['Is_Bottom_Quintile'].value_counts().values, 
                       labels=['Top 80%', 'Bottom 20%'], autopct='%1.1f%%')
        axes[0, 0].set_title('Target Variable Distribution')
        
        # Feature distributions for key variables
        key_features = ['Tot_Ret_1Y', 'Expense_Ratio', 'Fund_Age_Months', 'Log_AUM']
        for i, feature in enumerate(key_features):
            if feature in self.df.columns:
                ax = axes[0, 1] if i == 0 else axes[0, 2] if i == 1 else axes[1, 0] if i == 2 else axes[1, 1]
                
                # Box plot by target
                self.df.boxplot(column=feature, by='Is_Bottom_Quintile', ax=ax)
                ax.set_title(f'{feature} by Target')
                ax.set_xlabel('Is Bottom Quintile')
        
        # Correlation heatmap
        ax_corr = axes[1, 2]
        corr_features = [f for f in feature_names[:10] if f in self.df.columns]  # Top 10 for readability
        corr_matrix = self.df[corr_features + ['Is_Bottom_Quintile']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax_corr, 
                   fmt='.2f', square=True)
        ax_corr.set_title('Feature Correlation Matrix')
        
        plt.tight_layout()
        plt.show()
        
        # 3. Model Diagnostics
        best_result = results[best_model_name]
        
        fig, axes = plt.subplots(2, 2, figsize=fig_size)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, best_result['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title(f'Confusion Matrix - {best_model_name}')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # Prediction Probability Distribution
        axes[0, 1].hist(best_result['y_prob'][y_test == 0], alpha=0.5, label='Class 0', bins=30)
        axes[0, 1].hist(best_result['y_prob'][y_test == 1], alpha=0.5, label='Class 1', bins=30)
        axes[0, 1].set_xlabel('Prediction Probability')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Prediction Probability Distribution')
        axes[0, 1].legend()
        
        # Calibration Curve
        from sklearn.calibration import calibration_curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, best_result['y_prob'], n_bins=10)
        axes[1, 0].plot(mean_predicted_value, fraction_of_positives, "s-", label=best_model_name)
        axes[1, 0].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        axes[1, 0].set_xlabel('Mean Predicted Probability')
        axes[1, 0].set_ylabel('Fraction of Positives')
        axes[1, 0].set_title('Calibration Curve')
        axes[1, 0].legend()
        
        # Learning Curve
        from sklearn.model_selection import learning_curve
        train_sizes, train_scores, val_scores = learning_curve(
            best_model, X_train, y_train, cv=5, scoring='roc_auc',
            train_sizes=np.linspace(0.1, 1.0, 10))
        
        axes[1, 1].plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training')
        axes[1, 1].plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label='Validation')
        axes[1, 1].fill_between(train_sizes, np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                               np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.1)
        axes[1, 1].fill_between(train_sizes, np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                               np.mean(val_scores, axis=1) + np.std(val_scores, axis=1), alpha=0.1)
        axes[1, 1].set_xlabel('Training Set Size')
        axes[1, 1].set_ylabel('ROC AUC Score')
        axes[1, 1].set_title('Learning Curve')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
        
        return best_model_name, best_result
    
    def run_complete_analysis(self):
        """Run the complete enhanced analysis pipeline"""
        print("Starting Enhanced Model Analysis...")
        
        # Step 1: Create advanced features
        self.create_advanced_features()
        
        # Step 2: Prepare data
        X, y, feature_names = self.prepare_data()
        
        # Step 3: Train-test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SET_SIZE, 
            random_state=config.RANDOM_STATE, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Class balance - Train: {y_train.value_counts(normalize=True)}")
        print(f"Class balance - Test: {y_test.value_counts(normalize=True)}")
        
        # Step 4: Train multiple models
        results = self.train_multiple_models(X_train, X_test, y_train, y_test)
        
        # Step 5: Create comprehensive visualizations
        best_model_name, best_result = self.create_comprehensive_visualizations(
            X_train, X_test, y_train, y_test, results, feature_names
        )
        
        # Step 6: Print detailed results
        print(f"\n{'='*50}")
        print("FINAL RESULTS SUMMARY")
        print(f"{'='*50}")
        
        for name, result in results.items():
            print(f"\n{name}:")
            print(f"  ROC-AUC: {result['roc_auc']:.4f}")
            print(f"  Precision: {result['precision']:.4f}")
            print(f"  Recall: {result['recall']:.4f}")
            print(f"  F1-Score: {result['f1']:.4f}")
            print(f"  Best Parameters: {result['best_params']}")
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Best ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")
        
        return results, best_model_name


def train_and_evaluate_enhanced(df: pd.DataFrame):
    """Enhanced version of the original train_and_evaluate function"""
    if df.empty:
        print("Dataframe is empty. Halting model training.")
        return

    trainer = EnhancedModelTrainer(df)
    results, best_model_name = trainer.run_complete_analysis()
    
    return results[best_model_name]['model']