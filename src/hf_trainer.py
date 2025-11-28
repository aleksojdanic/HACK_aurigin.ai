import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from hf_utils import load_data_hf


def train_random_forest():
    print("Loading HF dataset...")
    X_train, y_train, X_val, y_val, X_test, test_split = load_data_hf(
        limit_train=None,  # kannst erhöhen
        limit_val= None,
        limit_test= None
    )

    print("Shapes:")
    print("X_train:", X_train.shape)
    print("X_val:  ", X_val.shape)
    print("X_test: ", X_test.shape)

    print("\nTraining RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample",
    )

    rf.fit(X_train, y_train)

    print("\nEvaluating...")
    y_pred = rf.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    print(f"\nAccuracy (val): {acc:.4f}\n")
    print(classification_report(y_val, y_pred))

    return rf, X_test, test_split


import pandas as pd


def create_submission_csv(model, X_test, test_split, out_path="submission.csv"):
    """
    Erzeugt eine Submission-CSV:
    - id
    - prediction
    """

    # Vorhersagen (Array mit Labels)
    y_pred = model.predict(X_test)

    # IDs aus HuggingFace-Split
    ids = [row["id"] for row in test_split]

    # DataFrame bauen
    df = pd.DataFrame({
        "id": ids,
        "prediction": y_pred.astype(int)
    })

    # CSV speichern
    df.to_csv(out_path, index=False)
    print(f"Submission saved to: {out_path}")
