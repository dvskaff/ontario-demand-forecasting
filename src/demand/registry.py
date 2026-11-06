"""Model registry on MLflow (SQLite backend, local artifacts): every retrain is a logged run and a registered model
version; the `champion` alias marks the version in production and `previous` the one it replaced."""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

for k, v in {
    "MLFLOW_DISABLE_TELEMETRY": "true",
    "DO_NOT_TRACK": "1",
    "MLFLOW_DISABLE_AGENT_HINT": "1",
    "MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR": "false",
}.items():
    os.environ.setdefault(k, v)

import mlflow  # noqa: E402
import mlflow.pyfunc  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402

from .config import Config  # noqa: E402

log = logging.getLogger(__name__)
EXPERIMENT = "ontario-demand"


class QuantileOnnxModel(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper: load with mlflow.pyfunc.load_model(...) and predict on a feature DataFrame."""

    def load_context(self, context):
        from demand.model import OnnxModels

        self.models = OnnxModels.from_dir(Path(context.artifacts["onnx"]))

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        return self.models.predict(model_input)


def client(cfg: Config) -> MlflowClient:
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(cfg.mlflow_uri)
    c = MlflowClient()
    if c.get_experiment_by_name(EXPERIMENT) is None:
        c.create_experiment(EXPERIMENT, artifact_location=cfg.artifact_root.as_uri())
    mlflow.set_experiment(EXPERIMENT)
    return c


def log_version(
    cfg: Config, onnx_dir: Path, params: dict, metrics: dict, tags: dict, files: dict[str, Path]
) -> str:
    """Log a training run (params, metrics, tags, files) and register its model. Returns the new version number."""
    client(cfg)
    with mlflow.start_run(run_name=f"retrain {tags.get('data_end', '')}") as run:
        mlflow.log_params(params)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})
        mlflow.set_tags(tags)
        for name, path in files.items():
            mlflow.log_artifact(str(path), artifact_path=name)
        info = mlflow.pyfunc.log_model(name="model", python_model=QuantileOnnxModel(),
                                       artifacts={"onnx": str(onnx_dir)},
                                       pip_requirements=["onnxruntime", "pandas", "numpy"])  # fmt: skip
        mv = mlflow.register_model(info.model_uri, cfg.model_name, tags=tags)
    log.info("Registered %s version %s (run %s)", cfg.model_name, mv.version, run.info.run_id)
    return str(mv.version)


def champion(cfg: Config) -> dict | None:
    c = client(cfg)
    try:
        v = c.get_model_version_by_alias(cfg.model_name, "champion")
    except MlflowException:
        return None
    run = c.get_run(v.run_id)
    local = Path(mlflow.artifacts.download_artifacts(f"models:/{cfg.model_name}/{v.version}"))
    return {"version": str(v.version), "run_id": v.run_id, "tags": dict(v.tags), "metrics": run.data.metrics,
            "params": run.data.params, "onnx_dir": local / "artifacts" / "onnx", "created": v.creation_timestamp}  # fmt: skip


def promote(cfg: Config, version: str) -> None:
    c = client(cfg)
    old = champion(cfg)
    if old:
        c.set_registered_model_alias(cfg.model_name, "previous", old["version"])
        c.set_model_version_tag(cfg.model_name, old["version"], "status", "retired")
    c.set_registered_model_alias(cfg.model_name, "champion", version)
    c.set_model_version_tag(cfg.model_name, version, "status", "champion")


def versions(cfg: Config) -> list[dict]:
    c = client(cfg)
    try:
        found = c.search_model_versions(f"name='{cfg.model_name}'")
    except MlflowException:
        return []
    out = []
    for v in sorted(found, key=lambda v: int(v.version), reverse=True):
        m = c.get_run(v.run_id).data.metrics
        out.append({"version": int(v.version), "created": pd.Timestamp(v.creation_timestamp, unit="ms").strftime("%Y-%m-%d %H:%M"),
                    "aliases": list(v.aliases), "metrics": m, **dict(v.tags)})  # fmt: skip
    return out


def prune(cfg: Config, keep: int = 26) -> int:
    """Delete old versions that are not champion/previous, with their files, so the saved state stays small.
    Only directories inside the project's artifact folder are ever removed."""
    c = client(cfg)
    root = cfg.artifact_root.resolve()
    removed = 0
    for v in versions(cfg)[keep:]:
        if v["aliases"]:
            continue
        mv = c.get_model_version(cfg.model_name, str(v["version"]))
        model_dir = Path(
            mlflow.artifacts.download_artifacts(f"models:/{cfg.model_name}/{v['version']}")
        ).resolve()
        c.delete_model_version(cfg.model_name, str(v["version"]))
        c.delete_run(mv.run_id)
        for d in (
            model_dir.parent,
            root / str(c.get_experiment_by_name(EXPERIMENT).experiment_id) / mv.run_id,
        ):
            if root in d.parents and d.exists():
                shutil.rmtree(d)
        removed += 1
    return removed


def write_json(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path
