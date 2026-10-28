import os
import time
import joblib
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime

from .data_agent import DataAgent
from .model_agent import ModelAgent
from .eval_agent import EvalAgent

class WorkflowOrchestrator:
    """Master agent that coordinates execution between Data, Model, and Eval agents."""

    def __init__(self, artifacts_dir: str = "saved_models"):
        self.name = "Orchestrator Agent"
        self.artifacts_dir = artifacts_dir
        self.data_agent = DataAgent()
        self.model_agent = ModelAgent()
        self.eval_agent = EvalAgent()
        self.execution_logs: List[Dict[str, Any]] = []

    def log(self, agent_name: str, step: str, message: str, status: str = "INFO"):
        entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "agent": agent_name,
            "step": step,
            "message": message,
            "status": status
        }
        self.execution_logs.append(entry)

    def run_pipeline(
        self,
        df: pd.DataFrame,
        target_col: str,
        task_type: str,
        algorithm: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        self.execution_logs = []
        self.log(self.name, "Pipeline Init", f"Starting ML Pipeline: Task='{task_type}', Model='{algorithm}', Target='{target_col}'")

        self.log(self.data_agent.name, "Data Inspection", f"Inspecting dataset with {len(df)} rows and {df.shape[1]} columns.")
        data_summary = self.data_agent.inspect_data(df, target_col=target_col)
        self.log(self.data_agent.name, "Data Clean and Split", f"Preprocessing dataset (test size: {test_size * 100}%).")

        split_data = self.data_agent.preprocess_and_split(
            df=df,
            target_col=target_col,
            test_size=test_size
        )
        self.log(
            self.data_agent.name,
            "Split Complete",
            f"Dataset split complete: Train samples = {len(split_data['X_train'])}, Test samples = {len(split_data['X_test'])}."
        )

        self.log(self.model_agent.name, "Model Training", f"Training {algorithm} with hyperparameters: {hyperparameters or {}}")
        start_train = time.time()
        trained_model = self.model_agent.train_model(
            task_type=task_type,
            algorithm=algorithm,
            X_train=split_data["X_train_scaled"],
            y_train=split_data["y_train"],
            hyperparameters=hyperparameters
        )
        train_duration = round(time.time() - start_train, 3)
        self.log(self.model_agent.name, "Training Finished", f"Model trained successfully in {train_duration}s.")

        self.log(self.eval_agent.name, "Model Evaluation", "Scoring test set and extracting feature importances...")
        eval_results = self.eval_agent.evaluate(
            task_type=task_type,
            model=trained_model,
            X_test=split_data["X_test_scaled"],
            y_test=split_data["y_test"],
            feature_names=split_data["feature_names"]
        )
        metrics_str = ", ".join([f"{k}: {v}" for k, v in eval_results["metrics"].items()])
        self.log(self.eval_agent.name, "Evaluation Complete", f"Metrics calculated: {metrics_str}")

        artifact_filename = f"{algorithm.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        artifact_path = os.path.join(self.artifacts_dir, artifact_filename)

        pipeline_metadata = {
            "algorithm": algorithm,
            "task_type": task_type,
            "target_col": target_col,
            "feature_names": split_data["feature_names"],
            "original_features": split_data["original_features"],
            "scaler": split_data["scaler"],
            "metrics": eval_results["metrics"],
            "timestamp": datetime.now().isoformat()
        }

        saved_path = self.model_agent.save_artifact(
            model=trained_model,
            pipeline_metadata=pipeline_metadata,
            filepath=artifact_path
        )
        self.log(self.name, "Artifact Export", f"Model bundle serialized to '{saved_path}'. Pipeline completed successfully!", status="SUCCESS")

        return {
            "model": trained_model,
            "metadata": pipeline_metadata,
            "data_summary": data_summary,
            "split_data": split_data,
            "eval_results": eval_results,
            "artifact_path": saved_path,
            "logs": self.execution_logs
        }

    @staticmethod
    def load_and_predict(artifact_path: str, input_df: pd.DataFrame) -> Any:
        bundle = joblib.load(artifact_path)
        model = bundle["model"]
        meta = bundle["metadata"]
        scaler = meta["scaler"]
        expected_features = meta["feature_names"]

        clean_input = input_df.copy()
        clean_input = pd.get_dummies(clean_input)
        for col in expected_features:
            if col not in clean_input.columns:
                clean_input[col] = 0
        clean_input = clean_input[expected_features]

        scaled_input = scaler.transform(clean_input)
        predictions = model.predict(scaled_input)
        return predictions
