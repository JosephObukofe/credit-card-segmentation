import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from credit_card_segmentation.data.custom import (
    load_data,
    preprocess_training_set,
    preprocess_test_set,
)
from credit_card_segmentation.config.config import (
    RAW_DATA,
    MODEL_TRAINING_DATA,
    EVALUATION_TEST_DATA,
)

df = load_data(RAW_DATA)
X_train, X_test = train_test_split(df, test_size=0.2)
X_train.drop(columns="CUST_ID", inplace=True)
X_test.drop(columns="CUST_ID", inplace=True)

sparse_skewed_features = [
    "PAYMENTS",
    "BALANCE",
    "PURCHASES",
    "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE",
    "ONEOFF_PURCHASES",
    "PURCHASES_TRX",
    "CASH_ADVANCE_TRX",
    "CASH_ADVANCE_FREQUENCY",
    "PRC_FULL_PAYMENT",
    "ONEOFF_PURCHASES_FREQUENCY",
    "BALANCE_FREQUENCY",
]

sparse_features = ["TENURE"]

skewed_features = [
    "PURCHASES_INSTALLMENTS_FREQUENCY",
    "PURCHASES_FREQUENCY",
]

missing_value_features = ["CREDIT_LIMIT"]

missing_value_skewed_features = ["MINIMUM_PAYMENTS"]

X_train_processed, preprocessor = preprocess_training_set(
    X_train,
    missing_value_features=missing_value_features,
    missing_value_skewed_features=missing_value_skewed_features,
    sparse_skewed_features=sparse_skewed_features,
    sparse_features=sparse_features,
    skewed_features=skewed_features,
)

X_test_processed = preprocess_test_set(
    X_test,
    preprocessor=preprocessor,
    missing_value_features=missing_value_features,
    missing_value_skewed_features=missing_value_skewed_features,
    sparse_skewed_features=sparse_skewed_features,
    sparse_features=sparse_features,
    skewed_features=skewed_features,
)

# Saving the preprocessed training and test sets for model training and evaluation
joblib.dump(X_train_processed, MODEL_TRAINING_DATA + "/X_train_processed.pkl")
joblib.dump(X_test_processed, EVALUATION_TEST_DATA + "/X_test_processed.pkl")
