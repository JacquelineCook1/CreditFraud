
Kaggle Credit Card Fraud Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

We used a pre-existing function to create our confusion matrices:
https://github.com/JacquelineCook1/CreditFraud/blob/e16c219a791b45d9e5d186acf4b58d0a0fb2b45b/randomForest.py#L45

This function plots the comparison between our F1, precision, and recall. We used matplotlib to generate the plots, and it takes the Precision, Recall, and F1 Score metrics from each model's test phase and groups them by metric category.
https://github.com/JacquelineCook1/CreditFraud/blob/95b885ad77a0188221c20c1b2bde94d1bd03b2ac/logisticRegression.py#L39-L80
