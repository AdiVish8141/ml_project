import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


class NaiveBaselineModel:
    """Benchmark Model: Predicts tomorrow's price using strictly today's latest observed price (delta = 0)."""
    def __init__(self):
        self.name = "Naive Baseline (Last-Observed Price)"
        
    def fit(self, X, y_delta):
        return self
        
    def predict(self, X):
        # Delta prediction is 0
        return np.zeros(len(X))

class PetrolModelSuite:
    """Manages models: Naive Baseline, Linear Regression, Ridge, Random Forest, XGBoost."""
    
    def __init__(self):
        self.models = {
            "Naive Baseline (Last Observed)": NaiveBaselineModel(),
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42),
            "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
        }
        if HAS_XGB:
            self.models["XGBoost Regressor"] = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
            
        self.trained_models = {}
        
    def train_all(self, X_train, y_train_delta):
        """Fits all models on the daily price change (delta)."""
        self.trained_models = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train_delta)
            self.trained_models[name] = model
        return self.trained_models

