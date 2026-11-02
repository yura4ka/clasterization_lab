"""Library-based UCI downloads, deterministic geometry and stochastic generators."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

from .artifacts import write_json


@dataclass
class Dataset:
    name: str
    frame: pd.DataFrame
    y: np.ndarray
    meta: dict = field(default_factory=dict)


UCI_DATASETS = {"wine": 109, "seeds": 236, "dry_bean": 602}
SHAPES = (
    "circles",
    "ellipses_parallel",
    "ellipses_rotated",
    "rectangles",
    "rings",
    "letters",
    "mixture",
)


def load_real(name, root):
    directory = Path(root) / "data" / "raw" / name
    directory.mkdir(parents=True, exist_ok=True)
    features = directory / "features.parquet"
    targets = directory / "targets.parquet"
    if features.exists() and targets.exists():
        frame, target = pd.read_parquet(features), pd.read_parquet(targets)
    else:
        if name == "seeds":
            # UCI id 236 exists but is not exposed through ucimlrepo's data API.
            os.environ["KAGGLEHUB_CACHE"] = str(
                (Path(root) / "data/raw/kaggle_cache").resolve()
            )
            import kagglehub

            cache = Path(kagglehub.dataset_download("rwzhang/seeds-dataset"))
            names = [
                "area",
                "perimeter",
                "compactness",
                "kernel_length",
                "kernel_width",
                "asymmetry",
                "groove_length",
                "class",
            ]
            raw = pd.read_csv(
                cache / "seeds_dataset.txt", sep=r"\s+", header=None, names=names
            )
            if raw.shape != (210, 8):
                raise ValueError(f"Unexpected Seeds schema: {raw.shape}")
            frame, target = raw.iloc[:, :7], raw[["class"]]
            write_json(
                directory / "source.json",
                {
                    "uci_id": 236,
                    "library": "kagglehub",
                    "handle": "rwzhang/seeds-dataset",
                    "resolved_cache": str(cache),
                    "reason": "Not available through ucimlrepo",
                },
            )
        else:
            data = fetch_ucirepo(id=UCI_DATASETS[name])
            frame, target = data.data.features.copy(), data.data.targets.copy()
            data.variables.to_csv(directory / "variables.csv", index=False)
            write_json(
                directory / "source.json",
                {
                    "uci_id": UCI_DATASETS[name],
                    "library": "ucimlrepo",
                    "metadata": dict(data.metadata),
                },
            )
        frame.to_parquet(features, index=False)
        target.to_parquet(targets, index=False)
    if not all(pd.api.types.is_numeric_dtype(t) for t in frame.dtypes):
        raise ValueError(f"{name}: nonnumeric feature detected")
    if target.shape[1] != 1 or target.isna().any().any():
        raise ValueError(f"{name}: expected a single complete evaluation target")
    labels, classes = pd.factorize(target.iloc[:, 0], sort=True)
    return Dataset(
        name,
        frame,
        labels,
        {
            "kind": "real",
            "uci_id": UCI_DATASETS[name],
            "classes": classes.tolist(),
            "target_role": "evaluation_only",
        },
    )


def _unit(n, rng):
    if rng is not None:
        return rng.random((n, 2))
    side = int(np.ceil(np.sqrt(n)))
    axis = (np.arange(side) + 0.5) / side
    u, v = np.meshgrid(axis, axis)
    return np.column_stack([u.ravel(), v.ravel()])[:n]


def _rotate(points, angle):
    c, s = np.cos(angle), np.sin(angle)
    return points @ np.array([[c, s], [-s, c]])


def geometry(shape, n_per_cluster=90, generator="deterministic", seed=42, noise=0):
    if generator not in {"deterministic", "PCG64", "MT19937"}:
        raise ValueError(generator)
    rng = (
        None
        if generator == "deterministic"
        else np.random.Generator(getattr(np.random, generator)(seed))
    )
    clusters = []
    for i in range(3):
        count = n_per_cluster
        if shape == "mixture":
            count = max(12, int(n_per_cluster * (0.5, 1, 1.5)[i]))
        uv = _unit(count, rng)
        angle = 2 * np.pi * uv[:, 1]
        disk = np.sqrt(uv[:, :1]) * np.column_stack([np.cos(angle), np.sin(angle)])
        if shape == "circles":
            points = disk * (0.7, 1.0, 1.25)[i] + (3.1 * i, 0)
        elif shape.startswith("ellipses"):
            points = disk * [1.4, 0.3]
            points = _rotate(points, 0 if shape.endswith("parallel") else i * np.pi / 3)
            points += [i * 3.5, 0]
        elif shape == "rectangles":
            points = _rotate((uv - 0.5) * [2.0, 0.65], i * np.pi / 5)
            points += [i * 3.0, 0]
        elif shape == "rings":
            radius = (0.7 + i * 0.75) + (uv[:, 0] - 0.5) * 0.17
            points = radius[:, None] * np.column_stack([np.cos(angle), np.sin(angle)])
        elif shape == "letters":
            # Three separate thick L shapes, not disconnected line segments.
            points = uv * [1.5, 1.5]
            horizontal = uv[:, 0] < 0.5
            points[horizontal, 0] = uv[horizontal, 0] * 3
            points[horizontal, 1] *= 0.2
            points[~horizontal, 0] = (uv[~horizontal, 0] - 0.5) * 0.6
            points = _rotate(points, i * np.pi / 4) + [i * 3.4, 0]
        elif shape == "mixture":
            if i == 0:
                points = disk * 0.45
            elif i == 1:
                points = _rotate(disk * [1.2, 0.3], np.pi / 4)
            else:
                theta = uv[:, 1] * 1.65 * np.pi
                radius = 0.8 + (uv[:, 0] - 0.5) * 0.2
                points = radius[:, None] * np.column_stack(
                    [np.cos(theta), np.sin(theta)]
                )
            points += [i * 2.7, 0]
        else:
            raise ValueError(shape)
        clusters.append(points)
    X = np.vstack(clusters)
    y = np.concatenate([np.full(len(points), i) for i, points in enumerate(clusters)])
    if noise:
        low, high = X.min(axis=0) - 0.4, X.max(axis=0) + 0.4
        background = low + _unit(noise, rng) * (high - low)
        X = np.vstack([X, background])
        y = np.r_[y, np.full(noise, -1)]
    name = (
        f"{shape}_{generator}_{seed}" if rng is not None else f"{shape}_deterministic"
    )
    return Dataset(
        name,
        pd.DataFrame(X, columns=["x", "y"]),
        y,
        {
            "kind": "synthetic",
            "shape": shape,
            "generator": generator,
            "seed": seed if rng is not None else None,
            "n_per_cluster": n_per_cluster,
            "background_points": noise,
        },
    )


def generate_suite(root, mode="full", seed=42):
    datasets = {}
    specifications = [("deterministic", seed), ("PCG64", seed), ("MT19937", seed)]
    if mode == "full":
        specifications.append(("PCG64", seed + 1))
    for shape in SHAPES:
        for generator, state in specifications:
            ds = geometry(
                shape,
                70 if mode == "quick" else 100,
                generator,
                state,
                noise=18 if shape == "mixture" else 0,
            )
            datasets[ds.name] = ds
            directory = Path(root) / "data" / "synthetic"
            directory.mkdir(parents=True, exist_ok=True)
            ds.frame.assign(target=ds.y).to_parquet(
                directory / f"{ds.name}.parquet", index=False
            )
            write_json(directory / f"{ds.name}.json", ds.meta)
    return datasets


def prepare(ds, max_samples, seed, store):
    indices = np.arange(len(ds.frame))
    if len(indices) > max_samples:
        indices, _ = train_test_split(
            indices, train_size=max_samples, stratify=ds.y, random_state=seed
        )
        indices.sort()
    frame, y = ds.frame.iloc[indices], ds.y[indices]
    if np.isinf(frame.to_numpy(dtype=float)).any():
        raise ValueError(f"{ds.name}: infinite values")
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    imputed = imputer.fit_transform(frame)
    # Preserve 2D shape geometry: common scalar rather than per-axis rescaling.
    if ds.meta["kind"] == "synthetic":
        center = imputed.mean(axis=0)
        common_scale = float(np.sqrt(np.mean(np.var(imputed, axis=0)))) or 1.0
        scale = np.full(imputed.shape[1], common_scale)
    else:
        scaler = StandardScaler().fit(imputed)
        center, scale = scaler.mean_, scaler.scale_
    X = (imputed - center) / scale
    record = {
        "name": ds.name,
        "X": X,
        "y": y,
        "indices": indices,
        "kind": ds.meta["kind"],
        "representation": "original_scaled",
        "source": ds.name,
        "columns": frame.columns.tolist(),
        "meta": ds.meta,
    }
    store.arrays(
        f"intermediates/{ds.name}/preprocessing.npz",
        indices=indices,
        imputer_statistics=imputer.statistics_,
        center=center,
        scale=scale,
    )
    directory = store.project / "data" / "processed" / store.root.name
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / f"{ds.name}.npz", X=X, y=y, indices=indices)
    store.arrays(f"intermediates/{ds.name}/prepared.npz", X=X, y=y, indices=indices)
    return record


def pca_record(record, store):
    pca = PCA(n_components=2, svd_solver="full").fit(record["X"])
    name = record["name"] + "_pca2"
    store.arrays(
        f"intermediates/{name}/pca.npz",
        components=pca.components_,
        mean=pca.mean_,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )
    output = {
        **record,
        "name": name,
        "X": pca.transform(record["X"]),
        "representation": "pca2",
        "columns": ["PC1", "PC2"],
    }
    store.arrays(
        f"intermediates/{name}/prepared.npz",
        X=output["X"],
        y=output["y"],
        indices=output["indices"],
    )
    return output
