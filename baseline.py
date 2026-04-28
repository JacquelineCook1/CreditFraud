import pandas as pd
from randomForest import run_random_forest
from logisticRegression import run_logistic_regression
#import numpy as np 
#from sklearn.model_selection import train_test_split

# Read in the dataset 
df = pd.read_csv('data/creditcard.csv')


#Print some summary stats
print("\nShape:", df.shape)
print("\nColomns:", df.columns.tolist())
#print("\nSummary Stats: ")
#print(df.describe())
#print(df.head())

from sklearn.model_selection import train_test_split

X = df.drop("Class", axis=1)
y = df["Class"]

# Splits data into 80% training and 20% testing
X_train, X_test, y_train, y_test, = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42,
    stratify=y
    )

# Logistic regression baseline
print("\n logistic regression baseline:")
run_logistic_regression(X_train, y_train, X_test, y_test)

# Random forest baseline
print("\n random forest baseline:")
run_random_forest(X_train, y_train, X_test, y_test)