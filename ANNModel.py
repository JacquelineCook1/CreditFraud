import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

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

def build_ann_model(X_train, X_test, y_train, y_test):
    #print('printing til here')
    #Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

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
    model = Sequential([
        Dense(32, activation="relu", input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

    #Compile model
    model.compile(optimizer="adam", 
                  metrics=["recall", "precision", "f1_score"], 
                  loss="binary_crossentropy")
    
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    model.fit(X_train_scaled,
        y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=5, # change to 100 epochs to really test 
        batch_size=256,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=2) 
    
    return model, scaler

build_ann_model(X_train, X_test, y_train, y_test)

