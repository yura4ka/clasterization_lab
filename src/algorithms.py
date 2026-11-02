"""PAM, GMM, sparse MCL and an explicitly specified Haar WaveCluster variant.

WaveCluster reference: Sheikholeslami, Chatterjee, Zhang (VLDB 1998),
https://www.vldb.org/conf/1998/p428.pdf . This implementation uses decimated
Haar approximation coefficients, fixed relative significance and 8-connectivity.
It is NOT a numerical replication of the paper's experimental filter settings.

MCL reference: van Dongen (2000), Graph Clustering by Flow Simulation,
https://ir.cwi.nl/pub/9256 . Expansion=2; inflation, pruning and column
normalization are applied until convergence. Graph construction is explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import warnings

import kmedoids
import numpy as np
import pywt
from scipy import ndimage, sparse
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

ALGORITHMS = ("PAM", "WaveCluster", "GMM", "MCL")


@dataclass
class Result:
    labels: np.ndarray
    diagnostics: dict = field(default_factory=dict)
    arrays: dict = field(default_factory=dict)
    matrices: dict = field(default_factory=dict)


def validate_X(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(X) < 3 or not np.isfinite(X).all():
        raise ValueError("Expected a finite matrix with at least three rows")
    return X


def bounds_for(X):
    low, high = X.min(axis=0), X.max(axis=0)
    pad = np.maximum((high - low) * 0.08, 0.1)
    return np.column_stack([low - pad, high + pad])


def default_params(algorithm, X):
    if algorithm == "PAM":
        return {"k": 3, "init": "build", "max_iter": 100}
    if algorithm == "GMM":
        return {"k": 3, "covariance_type": "full", "n_init": 3, "max_iter": 300}
    if algorithm == "WaveCluster":
        return {
            "bins": 32,
            "level": 1,
            "significance": 0.08,
            "min_cluster_size": 5,
            "bounds": bounds_for(X).tolist(),
        }
    if algorithm == "MCL":
        return {
            "neighbors": 10,
            "inflation": 2.0,
            "prune": 1e-4,
            "max_iter": 100,
            "tolerance": 1e-6,
        }
    raise ValueError(algorithm)


def pam(X, params, seed):
    k = int(params["k"])
    if not 1 <= k <= len(X):
        raise ValueError("Invalid number of medoids")
    distances = pairwise_distances(X, metric="euclidean")
    result = kmedoids.pam(
        distances,
        k,
        init=params.get("init", "build"),
        max_iter=params.get("max_iter", 100),
        random_state=seed,
    )
    return Result(
        np.asarray(result.labels, dtype=int),
        {
            "loss": float(result.loss),
            "iterations": int(result.n_iter),
            "implementation": "kmedoids.pam (original PAM, not FasterPAM)",
        },
        {"medoid_indices": np.asarray(result.medoids), "medoids": X[result.medoids]},
    )


def gmm(X, params, seed):
    model = GaussianMixture(
        n_components=int(params["k"]),
        covariance_type=params.get("covariance_type", "full"),
        n_init=params.get("n_init", 3),
        max_iter=params.get("max_iter", 300),
        reg_covar=1e-6,
        random_state=seed,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        labels = model.fit_predict(X)
    return Result(
        labels,
        {
            "converged": bool(model.converged_),
            "iterations": int(model.n_iter_),
            "bic": float(model.bic(X)),
            "aic": float(model.aic(X)),
            "lower_bound": float(model.lower_bound_),
            "warnings": [str(w.message) for w in caught],
        },
        {
            "means": model.means_,
            "covariances": model.covariances_,
            "weights": model.weights_,
            "probabilities": model.predict_proba(X),
        },
    )


def wavecluster(X, params, seed=42):
    """Quantize -> Haar DWT approximation -> significant cells -> components.

    For Haar with bins divisible by 2**level, each coarse coefficient corresponds
    exactly to a block of 2**level fine bins along each dimension. Consequently
    integer division gives an unambiguous point-to-component lookup table.
    Low-significance cells and undersized components are labelled noise (-1).
    Fixed bounds are deliberately respected for delta perturbation experiments.
    """
    if X.shape[1] != 2:
        raise ValueError("This WaveCluster variant is restricted to two dimensions")
    bins, level = int(params["bins"]), int(params["level"])
    if level < 1 or bins < 4 or bins % (2**level):
        raise ValueError("bins must be divisible by 2**level")
    significance = float(params["significance"])
    if not 0 < significance <= 1:
        raise ValueError("significance must be in (0, 1]")
    bounds = np.asarray(params.get("bounds", bounds_for(X)), dtype=float)
    if bounds.shape != (2, 2) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Invalid grid bounds")
    hist, _ = np.histogramdd(X, bins=(bins, bins), range=bounds.tolist())
    coefficients = pywt.wavedecn(hist, "haar", mode="zero", level=level)
    approximation = coefficients[0]
    # Convert orthonormal coefficients to average occupancy per fine cell.
    density = np.maximum(approximation / (2**level), 0)
    cutoff = significance * float(density.max())
    significant = (density >= cutoff) & (density > 0)
    components, n_components = ndimage.label(significant, structure=np.ones((3, 3)))
    scaled = (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    inside = ((scaled >= 0) & (scaled <= 1)).all(axis=1)
    fine = np.clip(np.floor(scaled * bins).astype(int), 0, bins - 1)
    coarse = fine // (2**level)
    labels = components[coarse[:, 0], coarse[:, 1]].astype(int) - 1
    labels[~inside] = -1
    for label in np.unique(labels[labels >= 0]):
        if np.count_nonzero(labels == label) < params.get("min_cluster_size", 5):
            labels[labels == label] = -1
    valid = sorted(set(labels) - {-1})
    mapping = {old: new for new, old in enumerate(valid)}
    labels = np.array([mapping.get(int(label), -1) for label in labels])
    return Result(
        labels,
        {
            "implementation": "WaveCluster/Haar-DWT-2D (explicit research variant)",
            "wavelet": "haar",
            "boundary_mode": "zero",
            "connectivity": 8,
            "cutoff": cutoff,
            "components_before_filter": int(n_components),
            "out_of_bounds": int((~inside).sum()),
        },
        {
            "histogram": hist,
            "density": density,
            "significant": significant,
            "components": components,
            "bounds": bounds,
            "coarse_indices": coarse,
        },
    )


def graph_for(X, neighbors=10):
    neighbors = min(int(neighbors), len(X) - 1)
    if neighbors < 1:
        raise ValueError("neighbors must be positive")
    nn = NearestNeighbors(n_neighbors=neighbors + 1).fit(X)
    distances, indices = nn.kneighbors(X)
    # Explicitly remove self, also when zero-distance duplicates alter ordering.
    rows, cols, values = [], [], []
    positive = distances[distances > 0]
    sigma = float(np.median(positive)) if len(positive) else 1.0
    for i in range(len(X)):
        mask = indices[i] != i
        js, ds = indices[i][mask][:neighbors], distances[i][mask][:neighbors]
        rows.extend([i] * len(js))
        cols.extend(js.tolist())
        values.extend(np.exp(-0.5 * (ds / max(sigma, 1e-12)) ** 2).tolist())
    graph = sparse.csc_matrix((values, (rows, cols)), shape=(len(X), len(X)))
    graph = graph.maximum(graph.T).tocsc()
    graph.setdiag(1.0)
    graph.eliminate_zeros()
    return graph, sigma


def normalize(matrix):
    sums = np.asarray(matrix.sum(axis=0)).ravel()
    if np.any(sums <= 0):
        raise ValueError("Empty stochastic-matrix column")
    return (matrix @ sparse.diags(1.0 / sums)).tocsc()


def markov_cluster(graph, inflation=2.0, prune=1e-4, max_iter=100, tolerance=1e-6):
    if inflation <= 1 or not 0 <= prune < 1:
        raise ValueError("inflation must be > 1; prune must be in [0,1)")
    matrix = normalize(graph)
    residual = np.inf
    history = []
    for iteration in range(1, max_iter + 1):
        previous = matrix
        matrix = (matrix @ matrix).tocsc()  # Expansion=2.
        matrix.data **= inflation
        matrix = normalize(matrix)
        # Keep the maximum in every column, even if all entries are below prune.
        for j in range(matrix.shape[1]):
            start, end = matrix.indptr[j : j + 2]
            if end > start:
                values = matrix.data[start:end]
                keep = int(np.argmax(values))
                mask = values < prune
                mask[keep] = False
                values[mask] = 0
        matrix.eliminate_zeros()
        matrix = normalize(matrix)
        difference = matrix - previous
        residual = float(np.max(np.abs(difference.data))) if difference.nnz else 0.0
        history.append((iteration, residual, matrix.nnz))
        if residual < tolerance:
            break
    # Standard attractor rows define supports; duplicate supports are merged.
    csr = matrix.tocsr()
    attractors = np.flatnonzero(matrix.diagonal() > 1e-6)
    supports = {}
    for attractor in attractors:
        row = csr.getrow(int(attractor))
        support = tuple(row.indices[row.data > 1e-5])
        if support:
            supports.setdefault(support, int(attractor))
    labels = np.full(matrix.shape[0], -1, dtype=int)
    # At finite tolerance overlapping supports can occur: assign by largest flow.
    if supports:
        representatives = list(supports.values())
        strengths = matrix[representatives, :].toarray()
        best = strengths.argmax(axis=0)
        covered = strengths.max(axis=0) > 1e-5
        labels[covered] = best[covered]
    for point in np.flatnonzero(labels < 0):
        labels[point] = labels.max(initial=-1) + 1
    _, labels = np.unique(labels, return_inverse=True)
    return labels, matrix, np.asarray(history), residual


def mcl(X, params, seed=42):
    graph, sigma = graph_for(X, params.get("neighbors", 10))
    labels, matrix, history, residual = markov_cluster(
        graph,
        params.get("inflation", 2.0),
        params.get("prune", 1e-4),
        params.get("max_iter", 100),
        params.get("tolerance", 1e-6),
    )
    return Result(
        labels,
        {
            "iterations": len(history),
            "residual": residual,
            "converged": residual < params.get("tolerance", 1e-6),
            "sigma": sigma,
            "graph_nnz": graph.nnz,
            "graph": "symmetric union kNN, Gaussian weights, unit loops",
            "extraction": "unique attractor supports, maximum-flow overlap resolution",
        },
        {"history": history},
        {"graph": graph, "flow": matrix},
    )


def run_algorithm(algorithm, X, params=None, seed=42):
    X = validate_X(X)
    params = default_params(algorithm, X) if params is None else dict(params)
    implementations = {"PAM": pam, "WaveCluster": wavecluster, "GMM": gmm, "MCL": mcl}
    with threadpool_limits(limits=1):
        result = implementations[algorithm](X, params, seed)
    if len(result.labels) != len(X):
        raise RuntimeError("Algorithm returned wrong number of labels")
    return result
