import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def load_and_prepare(path: str):
    df = pd.read_csv(path)
    # Drop first three non-informative columns + separate target
    X_df = df.drop(columns=["Number", "Customer_ID", "Last_Name", "Acc_Closed"])
    y = df["Acc_Closed"].astype(int).values
    # Categorical and numeric features
    categorical_cols = ["Location", "Gender"]
    numeric_cols = [c for c in X_df.columns if c not in categorical_cols]
    return X_df, y, categorical_cols, numeric_cols

def preprocess(X_train_df, X_test_df, categorical_cols, numeric_cols):
    ct = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ]
    )
    X_train = ct.fit_transform(X_train_df)
    X_test = ct.transform(X_test_df)
    return X_train, X_test, ct

def build_model(input_dim: int):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def main(data_path: str = "Bank_Predictions.csv",
         epochs: int = 50,
         batch_size: int = 64,
         random_state: int = 42):
    # Load
    X_df, y, categorical_cols, numeric_cols = load_and_prepare(data_path)
    # Split
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=random_state, stratify=y
    )
    # Preprocess
    X_train, X_test, ct = preprocess(X_train_df, X_test_df, categorical_cols, numeric_cols)
    # Build and train
    model = build_model(X_train.shape[1])
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    # Evaluate
    y_pred_prob = model.predict(X_test).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    cm = confusion_matrix(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print('Confusion Matrix:\n', cm)
    print('\nClassification Report:\n', classification_report(y_test, y_pred, digits=3))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC AUC = {auc:.3f}')
    plt.plot([0,1],[0,1],'--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Account Closure Prediction (Keras)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
