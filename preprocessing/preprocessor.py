import joblib
from sklearn.preprocessing import StandardScaler


class Preprocessor:

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X):
        self.scaler.fit(X)
        self.is_fitted = True
        print("[Preprocessor] Fitted scaler on training data.")

    def transform(self, X):
        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted.")
        return self.scaler.transform(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def save(self, path="artifacts/scaler.pkl"):
        joblib.dump(self.scaler, path)
        print(f"[Preprocessor] Scaler saved to {path}")

    def load(self, path="artifacts/scaler.pkl"):
        self.scaler = joblib.load(path)
        self.is_fitted = True
        print(f"[Preprocessor] Scaler loaded from {path}")