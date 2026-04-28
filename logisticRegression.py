import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

def run_logistic_regression(X_train, y_train, X_test, y_test):

    # Train the logistic regression model
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # calculating metrics (precision, recall, f1)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # printing results
    print("Logestic Regression Results: ")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    return precision, recall, f1

