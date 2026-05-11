from xml.parsers.expat import model

import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

#Import KaggleHub to load dataset
import kagglehub
from kagglehub import KaggleDatasetAdapter
import os

"""" Code used when not running from main.py
#Import dataset from KaggleHub
df = kagglehub.load_dataset(
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

"""

#---------------------------------------------------------------------------------------
# build ANN model
def build_ann_model(X,y):
    #Train/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y, 
    test_size=0.3, 
    random_state=42, 
    stratify=y
    )

    #Val / test split 
    X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

    #print('printing til here')
    #Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    X_val_scaled   = scaler.transform(X_val) # using test set as validation set for early stopping

    #Compute class weights
    classes = np.array([0, 1])
    weights = compute_class_weight( 
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weights = {
        0: weights[0], #not fruad yay
        1: weights[1]  #yucky scary scary fruad ahh
    }

    print("Class weights:", class_weights)

    #Build ANN
    # Use batch normalization to stabalize and dropout to help with overfitting, 
    # given the small dataset size and class imbalance
    # Final sigmoid layer outputs a probability between 0 and 1 representing fraud likelihood. 1=fraud
    model = Sequential([
         tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),

         tf.keras.layers.Dense(64, activation="relu"),
         tf.keras.layers.BatchNormalization(),
         tf.keras.layers.Dropout(0.3),

         tf.keras.layers.Dense(32, activation="relu"),
         tf.keras.layers.BatchNormalization(),
         tf.keras.layers.Dropout(0.25),

         tf.keras.layers.Dense(16, activation="relu"),
         tf.keras.layers.Dropout(0.15),

         tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    #Compile model
    model.compile(optimizer="adam", 
                  metrics=[
                      tf.keras.metrics.Recall(name="recall"),
                      tf.keras.metrics.Precision(name="precision"),
                      tf.keras.metrics.AUC(curve="PR", name="pr_auc")
                      ], 
                  loss="binary_crossentropy")
    
    # Stop training when validation PR-AUC stops improving to reduce overfitting
    # and restore the best-performing model weights.
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=8,
        restore_best_weights=True
    )

    model.fit(X_train_scaled,
        y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=100,
        batch_size=512,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=2 
    )

    # Predict probabilities on validation set
    val_probs = model.predict(X_val_scaled).ravel()

    # Tune threshold using F1 score
    precision, recall, thresholds = precision_recall_curve(y_val, val_probs)

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)

    best_threshold = thresholds[best_idx]
    print("Best threshold:", best_threshold)
    print("Validation precision:", precision[best_idx])
    print("Validation recall:", recall[best_idx])
    print("Validation F1:", f1_scores[best_idx])

    # Final test evaluation
    test_probs = model.predict(X_test_scaled).ravel()
    test_preds = (test_probs >= best_threshold - 0.03).astype(int)

    print("Confusion Matrix Format: [[Legit Labeled Correct, Legit Flagged Fraud], [Missed Fraud, Caught Fraud]]")
    print(confusion_matrix(y_test, test_preds))
    print(classification_report(y_test, test_preds, digits=4))
    
    return model, scaler, precision[best_idx], recall[best_idx], f1_scores[best_idx]

#build_ann_model(X,y)  use for testing ANNModel.py independently, but when running from main.py we want to return the model and scaler so we can use them for the comparison plot

