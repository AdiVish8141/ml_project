import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

class EvalAgent:
    """Agent responsible for computing diagnostic performance metrics and diagnostics."""

    def __init__(self):
        self.name = "Evaluation Agent"
        self.role = "Model Validation, Scoring and Diagnostic Analytics"

    def evaluate(
        self,
        task_type: str,
        model: Any,
        X_test,
        y_test,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        y_pred = model.predict(X_test)
        results: Dict[str, Any] = {
            "task_type": task_type,
            "y_test": y_test.tolist() if isinstance(y_test, (pd.Series, np.ndarray)) else list(y_test),
            "y_pred": y_pred.tolist() if isinstance(y_pred, (pd.Series, np.ndarray)) else list(y_pred),
            "metrics": {},
            "feature_importance": {}
        }

        if task_type == "regression":
            mse = mean_squared_error(y_test, y_pred)
            rmse = float(np.sqrt(mse))
            mae = float(mean_absolute_error(y_test, y_pred))
            r2 = float(r2_score(y_test, y_pred))

            results["metrics"] = {
                "R2 Score": round(r2, 4),
                "RMSE": round(rmse, 4),
                "MAE": round(mae, 4),
                "MSE": round(float(mse), 4)
            }
        else:
            acc = float(accuracy_score(y_test, y_pred))
            prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

            unique_labels = sorted(list(set(y_test).union(set(y_pred))))
            cm = confusion_matrix(y_test, y_pred, labels=unique_labels).tolist()

            results["metrics"] = {
                "Accuracy": round(acc, 4),
                "Precision (Weighted)": round(float(prec), 4),
                "Recall (Weighted)": round(float(rec), 4),
                "F1 Score (Weighted)": round(float(f1), 4)
            }
            results["confusion_matrix"] = {
                "matrix": cm,
                "labels": [str(lbl) for lbl in unique_labels]
            }
            results["classification_report"] = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feat_imp = {feat: float(round(imp, 4)) for feat, imp in zip(feature_names, importances)}
            results["feature_importance"] = dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True))
        elif hasattr(model, "coef_"):
            coef = model.coef_
            if coef.ndim > 1:
                coef = np.mean(np.abs(coef), axis=0)
            feat_imp = {feat: float(round(abs(c), 4)) for feat, c in zip(feature_names, coef)}
            results["feature_importance"] = dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True))

        return results
