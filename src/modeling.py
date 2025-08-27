# src/modeling.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import config

def train_and_evaluate(df: pd.DataFrame):
    """
    Splits data, trains a Random Forest model, and evaluates its performance
    on the test set.
    """
    if df.empty:
        print("Dataframe is empty. Halting model training.")
        return

    print("\nStep 4: Model Training and Evaluation...")

    # Define predictors (X) and target (y)
    # These features are selected based on the research plan and data availability.
    predictors = [
        'Fund_Age_Months', 'Log_AUM', 'Expense_Ratio', 'Tot_Ret_1M',
        'Tot_Ret_3M', 'Tot_Ret_6M', 'Tot_Ret_1Y', 'Std_Dev_1Y-M', 'Beta_3Y',
        'Avg_Dvd_Yield', 'Alpha_3Y'
    ]
    
    # Ensure all selected predictors exist in the dataframe
    predictors = [p for p in predictors if p in df.columns]
    print(f"\nUsing the following predictors for the model:\n{predictors}")
    
    X = df[predictors]
    y = df['Is_Bottom_Quintile']

    # --- Train-Test Split ---
    # We use 'stratify=y' to ensure the test set has the same percentage of
    # underperforming funds as the full dataset.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SET_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    print(f"\nData split into {len(X_train)} training and {len(X_test)} testing samples.")

    # --- Model Training ---
    model = RandomForestClassifier(
        n_estimators=150,         # A reasonable number of trees for robustness.
        max_depth=12,             # Prevents overfitting by limiting tree depth.
        min_samples_leaf=10,      # Prevents learning from noise in the data.
        class_weight='balanced',  # Crucial for imbalanced target variable.
        random_state=config.RANDOM_STATE,
        n_jobs=-1                 # Use all available CPU cores.
    )
    model.fit(X_train, y_train)
    print("Random Forest model trained successfully.")

    # --- Evaluation on the held-out Test Set ---
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] # Probabilities for the '1' class

    print("\n--- Model Performance on Test Set ---")
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Precision (for 'Bottom Quintile' class): {precision_score(y_test, y_pred):.4f}")
    print(f"Recall (for 'Bottom Quintile' class): {recall_score(y_test, y_pred):.4f}")

    # --- Feature Importance ---
    feature_importances = pd.Series(model.feature_importances_, index=predictors).sort_values(ascending=False)
    print("\n--- Top 10 Feature Importances ---")
    print(feature_importances.head(10))

    # --- Visualization ---
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Predicted: Hold', 'Predicted: Switch-Out'])
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_yticklabels(['Actual: Hold', 'Actual: Switch-Out'], rotation=90, va='center')
    plt.title("Confusion Matrix on Test Data")
    plt.show()

    # 2. Feature Importance Plot
    plt.figure(figsize=(12, 7))
    feature_importances.plot(kind='bar')
    plt.title("Feature Importance from Random Forest Model")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.show()