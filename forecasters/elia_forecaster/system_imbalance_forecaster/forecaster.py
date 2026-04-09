import joblib
import numpy as np

class SystemImbalanceForecaster:
    """Charge le modèle .joblib et expose .predict(X) comme dans ton app BE validée."""
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)

    def predict(self, X: np.ndarray):
        return self.model.predict(X)
