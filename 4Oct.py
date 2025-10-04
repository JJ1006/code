import pandas as pd
import numpy as np

# Tools for splitting data and building preprocessing pipelines
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Metrics and plots to evaluate a binary classifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
)

import matplotlib.pyplot as plt

# TensorFlow / Keras for building the ANN
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def load_and_prepare(path: str):
    """
    Load the CSV and return:
      - X_df: features DataFrame (after dropping non-predictive columns and the target)
      - y: target array (Acc_Closed as 0/1)
      - categorical_cols: list of categorical feature names
      - numeric_cols: list of numeric feature names

    According to the assignment, 'Number', 'Customer_ID', and 'Last_Name' must be ignored.
    We also remove 'Acc_Closed' from features because it's the target.
    """
    df = pd.read_csv(path)

    # Drop the first three non-informative columns + the target to get X
    X_df = df.drop(columns=["Number", "Customer_ID", "Last_Name", "Acc_Closed"])

    # Target as integers (0 = kept, 1 = closed)
    y = df["Acc_Closed"].astype(int).values

    # These two are categorical in the dataset and need one-hot encoding
    categorical_cols = ["Location", "Gender"]

    # Everything else will be treated as numeric and scaled
    numeric_cols = [c for c in X_df.columns if c not in categorical_cols]

    return X_df, y, categorical_cols, numeric_cols


def preprocess(X_train_df, X_test_df, categorical_cols, numeric_cols):
    """
    Build a preprocessing pipeline that:
      - One-hot encodes categorical features (drop='first' avoids dummy variable trap)
      - Standardizes numeric features (mean=0, std=1)

    We fit the transformer on the training set and apply the same transformation to the test set.
    """
    ct = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ]
    )

    # Fit on training data only (to avoid data leakage), then transform both
    X_train = ct.fit_transform(X_train_df)
    X_test = ct.transform(X_test_df)

    return X_train, X_test, ct


def build_model(input_dim: int):
    """
    Define a simple feed-forward ANN for binary classification:
      - Two hidden layers with ReLU activation
      - Final layer with sigmoid to output a probability in [0,1]
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation="relu"),  # hidden layer 1
        layers.Dense(16, activation="relu"),  # hidden layer 2
        layers.Dense(1, activation="sigmoid"),  # output layer for binary classification
    ])

    # 'binary_crossentropy' is the standard loss for binary classification
    # 'adam' is a robust default optimizer
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main(
    data_path: str = "Bank_Predictions.csv",
    epochs: int = 50,
    batch_size: int = 64,
    random_state: int = 42
):
    """
    Orchestrates the full pipeline:
      1) Load data and define feature types
      2) Split into train/test (stratify to preserve 0/1 balance)
      3) Preprocess (encode + scale)
      4) Build and train the ANN
      5) Evaluate (accuracy, ROC-AUC, confusion matrix, classification report)
      6) Plot ROC curve

    You can tune:
      - epochs (more = potentially better fit, but watch for overfitting)
      - batch_size
      - model architecture inside build_model()
    """

    # 1) Load data and separate features/target
    X_df, y, categorical_cols, numeric_cols = load_and_prepare(data_path)

    # 2) Train/test split (stratify keeps class ratios similar in both splits)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # 3) Preprocess features: one-hot for categorical + scale numeric
    X_train, X_test, ct = preprocess(X_train_df, X_test_df, categorical_cols, numeric_cols)

    # 4) Build and train the neural network
    model = build_model(X_train.shape[1])
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,  # hold out part of the training data to monitor overfitting
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # 5) Evaluate on the test set
    # Predict probabilities in [0,1]
    y_pred_prob = model.predict(X_test).ravel()

    # Convert probabilities to class labels using a 0.5 threshold
    # (You can lower this threshold if you want higher recall on "closed" accounts.)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # Basic metrics
    acc = accuracy_score(y_test, y_pred)          # overall correctness
    auc = roc_auc_score(y_test, y_pred_prob)      # threshold-independent performance
    cm = confusion_matrix(y_test, y_pred)         # TP/FP/FN/TN breakdown

    print(f"Test Accuracy: {acc:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print('Confusion Matrix:\n', cm)
    print('\nClassification Report:\n', classification_report(y_test, y_pred, digits=3))

    # 6) ROC curve visualization (True Positive Rate vs False Positive Rate)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], '--')  # diagonal = random classifier
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Account Closure Prediction (Keras)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # By default, looks for "Bank_Predictions.csv" in the current working directory.
    # You can override training length like:
    # main(epochs=100, batch_size=128)
    main()
