class ModelWithScaler():
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)