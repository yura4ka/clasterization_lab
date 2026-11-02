"""Local, auditable storage for notebook outputs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


class Store:
    def __init__(self, root, mode):
        self.project = Path(root).resolve()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        self.root = self.project / "artifacts" / f"{stamp}_{mode}"
        for name in (
            "config",
            "environment",
            "eda",
            "labels",
            "intermediates",
            "metrics",
            "figures",
            "stability",
            "benchmarks",
            "logs",
        ):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(str(self.root))
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(
            self.root / "logs" / "execution.log", encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.logger.addHandler(handler)
        self.logger.propagate = False
        write_json(self.project / "artifacts" / "latest.json", {"run": str(self.root)})

    def json(self, name, value):
        write_json(self.root / name, value)

    def table(self, name, frame):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path.with_suffix(".csv"), index=False)
        frame.to_parquet(path.with_suffix(".parquet"), index=False)
        return frame

    def arrays(self, name, **arrays):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)

    def figure(self, name, fig, show=True):
        path = self.root / "figures" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("png", "svg"):
            fig.savefig(path.with_suffix(f".{suffix}"), dpi=130, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    def environment(self):
        packages = {
            d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
        }
        self.json(
            "environment/versions.json",
            {
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
                "packages": packages,
            },
        )
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        )
        (self.root / "environment" / "pip-freeze.txt").write_text(freeze)
        return packages

    def finish(self):
        entries = []
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                entries.append(
                    {
                        "path": str(path.relative_to(self.root)),
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
        self.json("manifest.json", {"artifacts": entries, "count": len(entries)})
        return len(entries)


def frame_info(frame):
    buffer = io.StringIO()
    frame.info(buf=buffer)
    return buffer.getvalue()


DIRECT_DEPENDENCIES = [
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "PyWavelets",
    "ucimlrepo",
    "jupytext",
    "nbclient",
    "nbformat",
    "ipykernel",
    "psutil",
    "pyarrow",
    "kmedoids",
    "threadpoolctl",
    "kagglehub",
]


def write_requirements(project):
    lines = ["# Tested with Python 3.14; exact direct dependencies."]
    lines += [
        f"{name}=={importlib.metadata.version(name)}" for name in DIRECT_DEPENDENCIES
    ]
    path = Path(project) / "requirements.txt"
    path.write_text("\n".join(lines) + "\n")
    return path
