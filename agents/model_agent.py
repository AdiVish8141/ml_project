import os
import joblib
from typing import Dict, Any
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

class ModelAgent:
    """Agent responsible for instantiating, tuning, training, and serializing ML models."""

    def __init__(self):
        self.name = "Model Agent"
        self.role = "Model Selection, Hyperparameter Tuning and Training"

    def get_supported_algorithms(self, task_type: str) -> Dict[str, str]:
        if task_type == "regression":
            return {
                "Random Forest Regressor": "Ensemble of decision trees, handles non-linear patterns well.",
                "Gradient Boosting Regressor": "Sequential boosting model, high predictive accuracy.",
                "Ridge Regression": "Linear regression with L2 regularization.",
                "Decision Tree Regressor": "Fast, interpretable single-tree baseline."
            }
        else:
            return {
                "Random Forest Classifier": "Ensemble classifier, robust against overfitting.",
                "Gradient Boosting Classifier": "Boosted decision trees for high-performance classification.",
                "Logistic Regression": "Calibrated probabilistic linear classifier.",
                "Decision Tree Classifier": "Interpretable tree-based classification model."
            }

    def train_model(
        self,
        task_type: str,
        algorithm: str,
        X_train,
        y_train,
        hyperparameters: Dict[str, Any] = None
    ) -> Any:
        params = hyperparameters or {}

        if task_type == "regression":
            if algorithm == "Random Forest Regressor":
                n_estimators = int(params.get("n_estimators", 100))
                max_depth = int(params.get("max_depth", 10)) if params.get("max_depth") else None
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            elif algorithm == "Gradient Boosting Regressor":
                n_estimators = int(params.get("n_estimators", 100))
                learning_rate = float(params.get("learning_rate", 0.1))
                model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
            elif algorithm == "Decision Tree Regressor":
                max_depth = int(params.get("max_depth", 10)) if params.get("max_depth") else None
                model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
            else:
                alpha = float(params.get("alpha", 1.0))
                model = Ridge(alpha=alpha)
        else:
            if algorithm == "Random Forest Classifier":
                n_estimators = int(params.get("n_estimators", 100))
                max_depth = int(params.get("max_depth", 10)) if params.get("max_depth") else None
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            elif algorithm == "Gradient Boosting Classifier":
                n_estimators = int(params.get("n_estimators", 100))
                learning_rate = float(params.get("learning_rate", 0.1))
                model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
            elif algorithm == "Decision Tree Classifier":
                max_depth = int(params.get("max_depth", 10)) if params.get("max_depth") else None
                model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            else:
                c_val = float(params.get("C", 1.0))
                model = LogisticRegression(C=c_val, max_iter=1000, random_state=42)

        model.fit(X_train, y_train)
        return model

    def save_artifact(self, model, pipeline_metadata: Dict[str, Any], filepath: str) -> str:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        bundle = {
            "model": model,
            "metadata": pipeline_metadata
        }
        joblib.dump(bundle, filepath)
        return filepath
