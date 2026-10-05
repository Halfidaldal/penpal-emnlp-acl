"""Config access and data I/O.

The condition to process is `active_experiment` in config.yaml, or the
EXPERIMENT environment variable when set.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_project_root() -> Path:
    return PROJECT_ROOT


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def get_active_experiment() -> str:
    return os.environ.get("EXPERIMENT") or load_config()["active_experiment"]


def get_experiment_config(experiment: Optional[str] = None) -> dict:
    experiments = load_config()["experiments"]
    experiment = experiment or get_active_experiment()
    if experiment not in experiments:
        raise ValueError(f"Unknown experiment '{experiment}'; expected one of {list(experiments)}")
    return experiments[experiment]


def get_shared_config() -> dict:
    return load_config()["shared"]


def get_data_path(stage: str = "processed", experiment: Optional[str] = None) -> Path:
    return PROJECT_ROOT / get_experiment_config(experiment)[f"{stage}_dir"]


def _path(filename: str, stage: str, experiment: Optional[str], mkdir: bool = False) -> Path:
    path = get_data_path(stage, experiment) / filename
    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_csv(filename: str, stage: str = "processed", experiment: Optional[str] = None, **kwargs) -> pd.DataFrame:
    return pd.read_csv(_path(filename, stage, experiment), **kwargs)


def save_csv(df: pd.DataFrame, filename: str, stage: str = "processed", experiment: Optional[str] = None) -> None:
    path = _path(filename, stage, experiment, mkdir=True)
    df.to_csv(path, index=False)
    print(f"Saved {path.relative_to(PROJECT_ROOT)}")


def load_parquet(filename: str, stage: str = "processed", experiment: Optional[str] = None) -> pd.DataFrame:
    return pd.read_parquet(_path(filename, stage, experiment))


def save_parquet(df: pd.DataFrame, filename: str, stage: str = "processed", experiment: Optional[str] = None) -> None:
    path = _path(filename, stage, experiment, mkdir=True)
    df.to_parquet(path)
    print(f"Saved {path.relative_to(PROJECT_ROOT)}")


def save_npy(arr: np.ndarray, filename: str, stage: str = "processed", experiment: Optional[str] = None) -> None:
    path = _path(filename, stage, experiment, mkdir=True)
    np.save(path, arr)
    print(f"Saved {path.relative_to(PROJECT_ROOT)}")


# Exchange metadata is recomputed by cleaning, so the interim values win over
# whatever an older processed file carries.
_PREFER_INTERIM = {
    "analysis_turn", "complete_exchange", "starter", "starter_side",
    "starter_type", "author_1_type", "author_2_type",
}


def backfill_interaction_metadata(df: pd.DataFrame, experiment: Optional[str] = None) -> pd.DataFrame:
    """Join the interim interaction metadata onto a processed frame."""
    meta = load_csv("interaction_level_stories_filtered.csv", stage="interim", experiment=experiment)
    keys = [c for c in ["conversation_id", "turn", "interaction_count"] if c in df.columns and c in meta.columns]
    if not keys:
        raise ValueError("No shared join keys (conversation_id/turn/interaction_count)")

    left = df.copy()
    for key in {"turn", "interaction_count"} & set(keys):
        left[key] = pd.to_numeric(left[key], errors="coerce")
        meta[key] = pd.to_numeric(meta[key], errors="coerce")

    meta_cols = [c for c in meta.columns if c not in keys]
    meta = meta[keys + meta_cols].drop_duplicates(subset=keys, keep="last")
    merged = left.merge(meta, on=keys, how="left", suffixes=("", "__meta"))

    for col in meta_cols:
        meta_col = f"{col}__meta"
        if meta_col not in merged.columns:
            continue
        if col in _PREFER_INTERIM:
            merged[col] = merged[meta_col].where(merged[meta_col].notna(), merged[col])
        else:
            merged[col] = merged[col].where(merged[col].notna(), merged[meta_col])
        merged.drop(columns=meta_col, inplace=True)

    return merged
