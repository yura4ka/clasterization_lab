"""Fair parameter selection, persisted results and perturbation experiments."""

from __future__ import annotations

import copy
import json
import time
import traceback

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from .algorithms import ALGORITHMS, default_params, run_algorithm
from .artifacts import json_safe


def quality(X, y, labels, seed=42):
    mask = labels >= 0
    n_clusters = len(set(labels[mask]))
    output = {
        "n_clusters": n_clusters,
        "coverage": float(mask.mean()),
        "noise_fraction": float((~mask).mean()),
        "ari_all": float(adjusted_rand_score(y, labels)),
        "ami_all": float(adjusted_mutual_info_score(y, labels)),
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,
        "ari_covered": np.nan,
        "ami_covered": np.nan,
        "metric_note": "",
    }
    if mask.sum() >= 2:
        output["ari_covered"] = float(adjusted_rand_score(y[mask], labels[mask]))
        output["ami_covered"] = float(adjusted_mutual_info_score(y[mask], labels[mask]))
    if not 2 <= n_clusters < mask.sum():
        output["metric_note"] = "Internal metrics require 2..n-1 nonnoise clusters"
        return output
    try:
        output["silhouette"] = float(
            silhouette_score(
                X[mask],
                labels[mask],
                sample_size=min(600, int(mask.sum())),
                random_state=seed,
            )
        )
        output["davies_bouldin"] = float(davies_bouldin_score(X[mask], labels[mask]))
        output["calinski_harabasz"] = float(
            calinski_harabasz_score(X[mask], labels[mask])
        )
    except ValueError as error:
        output["metric_note"] = str(error)
    return output


def candidates(algorithm, X, mode):
    base = default_params(algorithm, X)
    if algorithm == "PAM":
        return [
            {**base, "k": k}
            for k in ([2, 3, 4] if mode == "quick" else [2, 3, 4, 5, 7])
        ]
    if algorithm == "GMM":
        return [
            {**base, "k": k, "covariance_type": covariance}
            for k in ([2, 3, 4] if mode == "quick" else [2, 3, 4, 5, 7])
            for covariance in (["full"] if mode == "quick" else ["full", "diag"])
        ]
    if algorithm == "MCL":
        return [
            {**base, "neighbors": neighbors, "inflation": inflation}
            for neighbors in ([10] if mode == "quick" else [6, 12])
            for inflation in (1.5, 2.0, 2.5)
        ]
    if algorithm == "WaveCluster":
        return [
            {**base, "bins": bins, "significance": significance}
            for bins in ([32] if mode == "quick" else [32, 64])
            for significance in (0.04, 0.10, 0.20)
        ]
    raise ValueError(algorithm)


def save_result(store, record, algorithm, result, params, prefix="labels"):
    stem = f"{prefix}/{record['name']}/{algorithm}"
    store.arrays(
        stem + ".npz",
        labels=result.labels,
        y=record["y"],
        source_indices=record["indices"],
    )
    store.json(stem + ".json", {"params": params, "diagnostics": result.diagnostics})
    if result.arrays:
        store.arrays(f"intermediates/{record['name']}/{algorithm}.npz", **result.arrays)
    for name, matrix in result.matrices.items():
        path = store.root / "intermediates" / record["name"] / f"{algorithm}_{name}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(path, matrix)


def tune(records, store, mode, seed=42):
    rows, selected = [], {}
    for name, record in records.items():
        X = record["X"]
        for algorithm in ALGORITHMS:
            if algorithm == "WaveCluster" and X.shape[1] != 2:
                rows.append(
                    {
                        "dataset": name,
                        "algorithm": algorithm,
                        "status": "skipped",
                        "reason": "WaveCluster/Haar implementation is 2D only",
                    }
                )
                continue
            group = []
            for index, params in enumerate(candidates(algorithm, X, mode)):
                row = {
                    "dataset": name,
                    "algorithm": algorithm,
                    "candidate": index,
                    "params": json.dumps(json_safe(params), sort_keys=True),
                    "status": "ok",
                    "reason": "",
                }
                try:
                    result = run_algorithm(algorithm, X, params, seed)
                    # Never pass reference labels into the tuning criterion.
                    internal = quality(
                        X, np.zeros(len(X), dtype=int), result.labels, seed
                    )
                    row.update(
                        {
                            k: v
                            for k, v in internal.items()
                            if not k.startswith(("ari", "ami"))
                        }
                    )
                    row["bic"] = result.diagnostics.get("bic", np.nan)
                    row["converged"] = result.diagnostics.get("converged", True)
                    store.arrays(
                        f"intermediates/parameter_search/{name}/{algorithm}_{index}.npz",
                        labels=result.labels,
                    )
                    store.json(
                        f"intermediates/parameter_search/{name}/{algorithm}_{index}.json",
                        {"params": params, "diagnostics": result.diagnostics},
                    )
                except Exception as error:
                    row.update(
                        status="error", reason=f"{type(error).__name__}: {error}"
                    )
                    store.logger.exception("Tuning failed: %s %s", name, algorithm)
                rows.append(row)
                group.append(row)
            successful = [
                r for r in group if r["status"] == "ok" and r.get("converged", True)
            ]
            eligible = [
                r
                for r in successful
                if np.isfinite(r.get("silhouette", np.nan))
                and r["coverage"] >= 0.8
                and 2 <= r["n_clusters"] <= 20
            ]
            if algorithm == "GMM" and successful:
                best = min(successful, key=lambda r: r["bic"])
                rule = "minimum BIC among converged candidates; no reference labels"
            elif eligible:
                best = max(eligible, key=lambda r: r["silhouette"])
                rule = "maximum silhouette with coverage>=0.8 and 2..20 clusters"
            elif successful:
                # Explicit fallback, not a hidden optimization using labels.
                best = successful[0]
                rule = (
                    "fallback: first successful candidate; no eligible internal score"
                )
            else:
                store.logger.error("No successful candidate for %s/%s", name, algorithm)
                continue
            selected[(name, algorithm)] = json.loads(best["params"])
            store.json(
                f"config/selected/{name}/{algorithm}.json",
                {
                    "params": selected[(name, algorithm)],
                    "selection_rule": rule,
                    "candidate": best["candidate"],
                },
            )
        store.logger.info("Tuning completed: %s", name)
    frame = pd.DataFrame(rows)
    store.table("metrics/parameter_search", frame)
    return selected, frame


def evaluate(records, selected, store, seed=42, kind=None):
    rows, outputs = [], {}
    for name, record in records.items():
        if kind is not None and record["kind"] != kind:
            continue
        outputs[name] = {}
        for algorithm in ALGORITHMS:
            row = {
                "dataset": name,
                "algorithm": algorithm,
                "kind": record["kind"],
                "representation": record["representation"],
                "n": len(record["X"]),
                "d": record["X"].shape[1],
                "status": "ok",
                "reason": "",
            }
            if (name, algorithm) not in selected:
                row.update(
                    status="skipped",
                    reason="No selected configuration (see parameter search)",
                )
            else:
                params = selected[(name, algorithm)]
                try:
                    start = time.perf_counter()
                    result = run_algorithm(algorithm, record["X"], params, seed)
                    row["elapsed_in_process_s"] = time.perf_counter() - start
                    row.update(quality(record["X"], record["y"], result.labels, seed))
                    row["converged"] = result.diagnostics.get("converged", True)
                    row["params"] = json.dumps(params, sort_keys=True)
                    outputs[name][algorithm] = result
                    save_result(store, record, algorithm, result, params)
                except Exception as error:
                    row.update(
                        status="error", reason=f"{type(error).__name__}: {error}"
                    )
                    store.logger.exception("Evaluation failed: %s/%s", name, algorithm)
            rows.append(row)
        store.logger.info("Evaluation completed: %s", name)
    return pd.DataFrame(rows), outputs


def aligned_change(a, b):
    """Optimal cluster relabeling, with the noise role (-1) fixed."""
    a, b = np.asarray(a), np.asarray(b)
    a_values, b_values = np.unique(a[a >= 0]), np.unique(b[b >= 0])
    correct = int(np.count_nonzero((a == -1) & (b == -1)))
    if len(a_values) and len(b_values):
        counts = np.array(
            [
                [np.count_nonzero((a == u) & (b == v)) for v in b_values]
                for u in a_values
            ]
        )
        rows, cols = linear_sum_assignment(counts, maximize=True)
        correct += int(counts[rows, cols].sum())
    return 1.0 - correct / len(a)


def characteristic_distance(X):
    # Median pairwise distance is robust to extreme nearest-neighbor gaps.
    sample = X[np.linspace(0, len(X) - 1, min(len(X), 400), dtype=int)]
    distances = pdist(sample)
    positive = distances[distances > 0]
    return float(np.median(positive)) if len(positive) else 1.0


def representative_points(X, labels):
    from sklearn.neighbors import NearestNeighbors

    centers = {label: X[labels == label].mean(axis=0) for label in set(labels) - {-1}}
    distances = np.array(
        [
            np.linalg.norm(point - centers[label]) if label >= 0 else np.inf
            for point, label in zip(X, labels)
        ]
    )
    output = {"interior": int(np.argmin(distances))}
    nn = NearestNeighbors(n_neighbors=min(20, len(X))).fit(X)
    neighbors = nn.kneighbors(X, return_distance=False)
    disagreement = np.mean(labels[neighbors] != labels[:, None], axis=1)
    output["boundary"] = int(np.argmax(disagreement + np.minimum(distances, 10) * 1e-6))
    if np.any(labels == -1):
        output["noise"] = int(np.flatnonzero(labels == -1)[0])
    else:
        # A density-based sparse point, explicitly not claimed to be ground-truth noise.
        d, _ = NearestNeighbors(n_neighbors=min(6, len(X))).fit(X).kneighbors(X)
        output["sparse"] = int(np.argmax(d[:, -1]))
    return output


def stability(records, selected, outputs, store, mode, seed=42):
    requested = ["ellipses_rotated_deterministic", "rings_PCG64_42", "wine_pca2"]
    names = [name for name in requested if name in records]
    relative_deltas = [0.001, 0.01, 0.05, 0.1]
    directions = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    rows, examples = [], []
    for name in names:
        record, X = records[name], records[name]["X"]
        scale = characteristic_distance(X)
        for algorithm, baseline in outputs[name].items():
            params = copy.deepcopy(selected[(name, algorithm)])
            points = representative_points(X, baseline.labels)
            for role, point in points.items():
                groups = {"single": np.array([point])}
                if mode == "full" and role == "boundary":
                    nearest = np.argsort(np.linalg.norm(X - X[point], axis=1))[:3]
                    groups["three_points"] = nearest
                for group_name, indices in groups.items():
                    for direction_index, direction in enumerate(directions):
                        for delta in relative_deltas:
                            moved = X.copy()
                            moved[indices] += direction * scale * delta
                            row = {
                                "dataset": name,
                                "algorithm": algorithm,
                                "role": role,
                                "point": point,
                                "moved_indices": json.dumps(indices.tolist()),
                                "group": group_name,
                                "direction": direction_index,
                                "delta_relative": delta,
                                "delta_absolute": scale * delta,
                                "characteristic_distance": scale,
                                "status": "ok",
                                "reason": "",
                            }
                            key = f"{name}/{algorithm}/{role}_{group_name}_{direction_index}_{delta}"
                            try:
                                result = run_algorithm(algorithm, moved, params, seed)
                                row.update(
                                    ari=adjusted_rand_score(
                                        baseline.labels, result.labels
                                    ),
                                    changed_fraction=aligned_change(
                                        baseline.labels, result.labels
                                    ),
                                    noise_transitions=int(
                                        np.count_nonzero(
                                            (baseline.labels == -1)
                                            != (result.labels == -1)
                                        )
                                    ),
                                    n_clusters_before=len(set(baseline.labels) - {-1}),
                                    n_clusters_after=len(set(result.labels) - {-1}),
                                )
                                unmoved = np.ones(len(X), dtype=bool)
                                unmoved[indices] = False
                                row["ari_unmoved"] = adjusted_rand_score(
                                    baseline.labels[unmoved], result.labels[unmoved]
                                )
                                store.arrays(
                                    f"stability/labels/{key}.npz",
                                    labels=result.labels,
                                    moved_indices=indices,
                                    displacement=direction * scale * delta,
                                )
                                if (
                                    group_name == "single"
                                    and delta == 0.1
                                    and direction_index == 0
                                ):
                                    examples.append(
                                        (
                                            record,
                                            algorithm,
                                            baseline.labels,
                                            result.labels,
                                            moved,
                                            indices,
                                            row["ari"],
                                        )
                                    )
                            except Exception as error:
                                row.update(status="error", reason=str(error))
                                store.logger.exception("Stability failed: %s", key)
                            rows.append(row)
        store.logger.info("Stability completed: %s", name)
    frame = pd.DataFrame(rows)
    store.table("stability/results", frame)
    store.json(
        "stability/protocol.json",
        {
            "deltas": relative_deltas,
            "distance": "median positive pairwise distance",
            "directions": directions,
            "preprocessing": "fixed, not refitted",
            "wavecluster_bounds": "fixed from baseline",
            "mcl_graph": "rebuilt from moved X",
            "seed": seed,
            "parameters": "fixed from baseline selection",
        },
    )
    return frame, examples


def randomness_check(records, selected, outputs, store, seed=42):
    rows = []
    name = "wine_pca2"
    record = records[name]
    for algorithm, baseline in outputs[name].items():
        for new_seed in (seed, seed + 1, seed + 2):
            result = run_algorithm(
                algorithm, record["X"], selected[(name, algorithm)], new_seed
            )
            rows.append(
                {
                    "dataset": name,
                    "algorithm": algorithm,
                    "seed": new_seed,
                    "ari_to_baseline": adjusted_rand_score(
                        baseline.labels, result.labels
                    ),
                }
            )
            store.arrays(
                f"stability/randomness/{algorithm}_{new_seed}.npz", labels=result.labels
            )
    frame = pd.DataFrame(rows)
    store.table("stability/randomness", frame)
    return frame
