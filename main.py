import pandas as pd
import os
from sklearn.model_selection import train_test_split
from logisticRegression import run_logistic_regression, plot_comparison
from randomForest import run_random_forest
from ANNModel import build_ann_model 

# Read in the dataset 
df = pd.read_csv('data/creditcard.csv')

X = df.drop("Class", axis=1)
y = df["Class"]

# Splits data into 80% training and 20% testing
X_train, X_test, y_train, y_test, = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42,
    stratify=y
    )

if __name__ == "__main__":
    # 1. Logistic Regression
    print("\n--- Running Logistic Regression ---")
    lr_metrics = run_logistic_regression(X_train, y_train, X_test, y_test)

    # 2. Random Forest
    print("\n--- Running Random Forest ---")
    rf_metrics = run_random_forest(X_train, y_train, X_test, y_test)

    # 3. ANN Model
    print("\n--- Running ANN Model ---")
    # NEED TO have build_ann_model return (p, r, f1) so we can use the cpmparison plot
    ann_model, ann_scaler = build_ann_model(X_train, X_test, y_train, y_test)
    
    # 4. Compare Baseline Models
    plot_comparison(lr_metrics, rf_metrics)