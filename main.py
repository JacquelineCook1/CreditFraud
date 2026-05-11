import pandas as pd
import os
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Import your functions from your other files
from logisticRegression import run_logistic_regression, plot_comparison
from randomForest import run_random_forest
from ANNModel import build_ann_model 

# 1. Setup and Data Loading
def load_data():
    # Download/Load dataset from KaggleHub
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    csv_path = os.path.join(path, "creditcard.csv")
    
    df = pd.read_csv(csv_path)
    
    # Optional: Save a local copy for backup
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/creditcard_sample.csv", index=False)
    
    print(f"Dataset loaded: {len(df)} rows")
    return df

if __name__ == "__main__":
    # Load data
    df = load_data()

    # Split the data into features and target
    X = df.drop('Class', axis=1)
    y = df['Class']

    # --- Step 1: Logistic Regression ---
    print("\n--- Running Logistic Regression ---")
    # This captures the (precision, recall, f1) tuple
    lr_metrics = run_logistic_regression(X, y)

    # --- Step 2: Random Forest ---
    print("\n--- Running Random Forest ---")
    # Make sure run_random_forest also returns (precision, recall, f1)
    rf_metrics = run_random_forest(X, y)

    # --- Step 3: ANN Model ---
    print("\n--- Running ANN Model ---")
    # Here we unpack the 5 items returned by your ANN function
    ann_model, ann_scaler, ann_p, ann_r, ann_f1 = build_ann_model(X, y)
    ann_metrics = (ann_p, ann_r, ann_f1)


    # --- Step 4: Compare Baseline Models ---
    print("\n--- Generating Comparison Plot ---")
    # This sends the collected results to the plotting function
    plot_comparison(lr_metrics, rf_metrics, ann_metrics)
    
    # Optional: If you want to compare RF vs ANN instead:
    # plot_comparison(rf_metrics, ann_metrics)
    
    print("\nAll tasks completed successfully.")