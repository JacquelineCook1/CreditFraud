import pandas as pd
import os
from logisticRegression import run_logistic_regression, plot_comparison
from randomForest import run_random_forest
from ANNModel import build_ann_model 

#Import KaggleHub to load dataset
import kagglehub
from kagglehub import KaggleDatasetAdapter
import os

#Import dataset from KaggleHub
df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "mlg-ulb/creditcardfraud",
    "creditcard.csv"
)

os.makedirs("data", exist_ok=True)
df.to_csv("data/creditcard_sample.csv", index=False)

print(f"Full file saved: {len(df)} rows")

# Read in the dataset
df = pd.read_csv("data/creditcard_sample.csv")

# Split the data into features and target
X = df.drop('Class', axis=1)
y = df['Class']

if __name__ == "__main__":
    # 1. Logistic Regression
    print("\n--- Running Logistic Regression ---")
    lr_metrics = run_logistic_regression(X,y)

    # 2. Random Forest
    print("\n--- Running Random Forest ---")
    rf_metrics = run_random_forest(X,y)

    # 3. ANN Model
    print("\n--- Running ANN Model ---")
    # NEED TO have build_ann_model return (p, r, f1) so we can use the cpmparison plot
    ann_model, ann_scaler, p, r, f1 = build_ann_model(X,y)
    
    # 4. Compare Baseline Models
    plot_comparison(lr_metrics, rf_metrics)