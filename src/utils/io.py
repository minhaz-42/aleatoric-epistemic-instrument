"""Run directories, and writers for the two artefact kinds the protocol emits.

Every measurement script writes a `records.csv` (one row per scored prediction, the raw material)
and a `summary.json` (the reduced numbers that documentation quotes). Keeping both means a claim in
a document can always be traced to the rows it came from.
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "experiments", "results")
FIGURES = os.path.join(ROOT, "figures")
TABLES = os.path.join(ROOT, "tables")


def run_dir(name: str) -> str:
    """Stable per-experiment directory. Re-running an experiment overwrites it, by design: a result
    is defined by its script and seed, so two runs that differ are a defect, not two data points."""
    path = os.path.join(RESULTS, name)
    os.makedirs(path, exist_ok=True)
    return path


def _plain(obj):
    """Make numpy scalars and arrays JSON-serialisable without changing their values."""
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def write_summary(name: str, payload: dict) -> str:
    path = os.path.join(run_dir(name), "summary.json")
    payload = dict(payload)
    payload["_written"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(path, "w") as fh:
        json.dump(_plain(payload), fh, indent=2, sort_keys=True)
    return path


def write_records(name: str, frame: pd.DataFrame, filename: str = "records.csv") -> str:
    path = os.path.join(run_dir(name), filename)
    frame.to_csv(path, index=False)
    return path


def read_summary(name: str) -> dict:
    with open(os.path.join(RESULTS, name, "summary.json")) as fh:
        return json.load(fh)
