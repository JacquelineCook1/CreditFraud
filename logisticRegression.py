import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from ANNModel import X_train, X_test, y_train, y_test

def run_logistic_regression(X_train, y_train, X_test, y_test):

    # Train the logistic regression model
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # calculating metrics (precision, recall, f1)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Logestic Regression Results: ")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    return precision, recall, f1

def plot_metrics(precision, recall, f1):
    metrics = ['Precision', 'Recall', 'F1 Score']
    values = [precision, recall, f1]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(metrics, values)
  
    plt.ylim(0, 1.0)
    plt.ylabel('Score')
    plt.title('Logistic Regression Performance')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 2), ha='center', va='bottom')

    plt.show()

    p, r, f = run_logistic_regression(X_train, y_train, X_test, y_test)
    plot_metrics(p, r, f)

def plot_comparison(lr_metrics, rf_metrics):
    labels = ['Precision', 'Recall', 'F1 Score']
    
    x = np.arange(len(labels))
    width = 0.35  

    fig, ax = plt.subplots(figsize=(10, 6))
  
    rects1 = ax.bar(x - width/2, lr_metrics, width, label='Logistic Regression', color='#4285F4')
    rects2 = ax.bar(x + width/2, rf_metrics, width, label='Random Forest', color='#34A853')

    # text for labels
    ax.set_ylabel('Scores')
    ax.set_title('Model Comparison: Credit Fraud Detection')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.1)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()