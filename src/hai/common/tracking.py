from __future__ import annotations

import os
import sys
from numbers import Real


def log_cli_run(
    run_name: str, result: dict, *, tags: dict[str, str], params: dict[str, object] | None = None
) -> dict[str, object]:
    """Best-effort MLflow logging for short CLI experiments.

    Research commands remain usable without MLflow, while configured local runs
    get a searchable run, scalar metrics, parameters, tags, and a JSON result.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return {"enabled": False, "reason": "MLFLOW_TRACKING_URI is not set"}
    try:
        import mlflow
    except ImportError:
        return {"enabled": False, "reason": "mlflow is not installed"}
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "heterogeneous-ai"))
        with mlflow.start_run(run_name=run_name):
            if params:
                mlflow.log_params({key: str(value) for key, value in params.items()})
            mlflow.set_tags(tags)
            metrics: dict[str, float] = {}

            def collect(values: dict, prefix: str = "") -> None:
                for key, value in values.items():
                    metric_key = f"{prefix}.{key}" if prefix else str(key)
                    if isinstance(value, Real) and not isinstance(value, bool):
                        metrics[metric_key[:250]] = float(value)
                    elif isinstance(value, dict):
                        collect(value, metric_key)

            collect(result)
            if metrics:
                mlflow.log_metrics(metrics)
            mlflow.log_dict(result, "result.json")
            run_id = mlflow.active_run().info.run_id
        return {"enabled": True, "run_id": run_id, "tracking_uri": tracking_uri}
    except Exception as exc:  # MLflow must not make a research command fail.
        print(f"MLflow logging warning: {exc}", file=sys.stderr)
        return {"enabled": False, "reason": str(exc)}
