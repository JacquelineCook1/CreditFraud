import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def run_logistic_regression(X, y):
    """
    Trains the model and returns the metrics as a tuple.
    """
    # Splits data into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Train the logistic regression model
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Calculating metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Logistic Regression Results: ")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Return these so main.py can capture them in 'lr_metrics'
    return precision, recall, f1

def plot_comparison(lr_metrics, rf_metrics, ann_metrics):
    labels = ['Precision', 'Recall', 'F1 Score']
    x = np.arange(len(labels))
    width = 0.25  # Thinner bars to fit three

    fig, ax = plt.subplots(figsize=(12, 6))
  
    # Centering the bars: One to the left, one in the middle, one to the right
    rects1 = ax.bar(x - width, lr_metrics, width, label='Logistic Regression', color='#4285F4')
    rects2 = ax.bar(x, rf_metrics, width, label='Random Forest', color='#34A853')
    rects3 = ax.bar(x + width, ann_metrics, width, label='ANN Model', color='#FBBC05')
    
    ax.set_ylabel('Scores')
    ax.set_title('Model Comparison: Credit Fraud Detection')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Added + rects3 here so the ANN values also show up on the graph!
    for rect in rects1 + rects2 + rects3:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()