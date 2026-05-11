from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from logisticRegression import run_logistic_regression, plot_comparison
from sklearn.model_selection import train_test_split

def run_random_forest(X,y):
    # Splits data into 80% training and 20% testing
    X_train, X_test, y_train, y_test, = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42,
    stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100, # Number of trees in the forest
        random_state=42, # For reproducibility

        #Since our dataset is extremely imbalanced (0.172% fraud), 
        #we need to set class_weight to 'balanced' to give more weight to the fraud cases
        class_weight='balanced' #Important for fraud detection
    )

    # training model + making predictions
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # calculating metrics (precision, recall, f1)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # printing results
    print("Random Forest Results: ")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    return precision, recall, f1

