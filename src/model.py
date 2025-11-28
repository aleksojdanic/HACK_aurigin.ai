from sklearn.linear_model import LogisticRegression
import numpy as np



class Model:
    def __init__(self, config=None):
        """
        Einfaches Modell-Gerüst mit Logistic Regression als Baseline.
        """
        self.config = config or {}
        # einfacher Klassifikator
        self.clf = LogisticRegression(
            max_iter=self.config.get("max_iter", 1000)
        )

    def fit(self, X, y):
        """
        Trainiert das Modell auf den Daten X, y.
        """
        print("Training model (LogisticRegression)...")
        self.clf.fit(X, y)

    def predict(self, X) -> np.ndarray:
        """
        Gibt binäre Vorhersagen (0 oder 1) zurück.
        """
        print("Predicting...")
        y_pred = self.clf.predict(X)
        return y_pred

