import io
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing, load_iris, load_wine, load_diabetes

class DataAgent:
    """Agent responsible for dataset ingestion, generation, inspection, cleaning, and preprocessing."""

    def __init__(self):
        self.name = "Data Agent"
        self.role = "Data Ingestion, Validation, Cleaning and Feature Transformation"

    def load_preset_dataset(self, preset_name: str) -> Tuple[pd.DataFrame, str, str]:
        if preset_name == "California Housing (Regression)":
            bunch = fetch_california_housing(as_frame=True)
            df = bunch.frame.copy()
            target_col = "MedHouseVal"
            task_type = "regression"
        elif preset_name == "Iris Species (Classification)":
            bunch = load_iris(as_frame=True)
            df = bunch.frame.copy()
            target_col = "target"
            df[target_col] = df[target_col].map({i: name for i, name in enumerate(bunch.target_names)})
            task_type = "classification"
        elif preset_name == "Wine Quality (Classification)":
            bunch = load_wine(as_frame=True)
            df = bunch.frame.copy()
            target_col = "target"
            df[target_col] = df[target_col].map({i: f"Class_{name}" for i, name in enumerate(bunch.target_names)})
            task_type = "classification"
        elif preset_name == "Diabetes Progression (Regression)":
            bunch = load_diabetes(as_frame=True)
            df = bunch.frame.copy()
            target_col = "target"
            task_type = "regression"
        elif preset_name == "Synthetic Customer Churn (Classification)":
            df = self._generate_synthetic_churn()
            target_col = "Churn"
            task_type = "classification"
        else:
            raise ValueError(f"Unknown preset dataset: {preset_name}")

        return df, target_col, task_type

    def _generate_synthetic_churn(self, n_samples: int = 1000) -> pd.DataFrame:
        np.random.seed(42)
        tenure = np.random.randint(1, 72, size=n_samples)
        monthly_charges = np.random.uniform(20.0, 120.0, size=n_samples)
        total_charges = tenure * monthly_charges + np.random.normal(0, 15, size=n_samples)
        total_charges = np.maximum(total_charges, monthly_charges)
        support_calls = np.random.poisson(lam=2.0, size=n_samples)
        age = np.random.randint(18, 75, size=n_samples)

        logit = (
            -1.5
            - 0.05 * tenure
            + 0.02 * monthly_charges
            + 0.45 * support_calls
            + 0.01 * (age - 40)
        )
        prob = 1 / (1 + np.exp(-logit))
        churn = (np.random.rand(n_samples) < prob).astype(int)
        churn_labels = ["Retained" if c == 0 else "Churned" for c in churn]

        df = pd.DataFrame({
            "TenureMonths": tenure,
            "MonthlyCharges": np.round(monthly_charges, 2),
            "TotalCharges": np.round(total_charges, 2),
            "CustomerServiceCalls": support_calls,
            "CustomerAge": age,
            "Churn": churn_labels
        })
        return df

    def inspect_data(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        num_rows, num_cols = df.shape
        missing_values = df.isnull().sum().to_dict()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        return {
            "num_rows": num_rows,
            "num_cols": num_cols,
            "feature_names": [c for c in df.columns if c != target_col],
            "target_col": target_col,
            "missing_values": missing_values,
            "total_missing": int(df.isnull().sum().sum()),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "head": df.head(5).to_dict(orient="records"),
            "describe": df.describe().round(3).to_dict()
        }

    def preprocess_and_split(
        self,
        df: pd.DataFrame,
        target_col: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        df_clean = df.copy()

        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                else:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col]

        cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if len(cat_features) > 0:
            X = pd.get_dummies(X, columns=cat_features, drop_first=True)

        feature_names = X.columns.tolist()

        stratify = None
        if not pd.api.types.is_numeric_dtype(y) or (y.nunique() < 10 and not isinstance(y.iloc[0], (float, np.floating))):
            stratify = y

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names, index=X_train.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_names, index=X_test.index)

        return {
            "X_train": X_train,
            "X_test": X_test,
            "X_train_scaled": X_train_scaled,
            "X_test_scaled": X_test_scaled,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": feature_names,
            "scaler": scaler,
            "original_features": [c for c in df.columns if c != target_col]
        }
