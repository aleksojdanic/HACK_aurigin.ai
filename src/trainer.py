

from model import Model
from utils import load_data

from pathlib import Path
from sklearn.metrics import f1_score


def train_pipeline():
    print("Starting training pipeline...")

    # 1. Daten laden – aktuell z.B. fake_train/ und fake_test/ Ordner
    X_train, y_train, X_test, y_test = load_data(
        train_dir=Path(r"HACK_aurigin.ai\Audio_tests\train_data_tests"),
        test_dir=Path(r"HACK_aurigin.ai\Audio_tests\test_data_tests"),
    )

    # 2. Modell initialisieren
    model = Model()

    # 3. Trainieren
    model.fit(X_train, y_train)

    print("Model training complete.")

    # 4. Evaluation auf Testdaten (mit Dummy-Labels noch sinnlos, aber Pipeline läuft)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)


    print(f"F1-Score (Dummy-Labels): {f1:.4f}")

    return model


