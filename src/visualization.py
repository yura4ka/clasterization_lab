"""EDA and publication-ready local figures (short captions live in the notebook)."""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse
import seaborn as sns
from sklearn.decomposition import PCA

from .artifacts import frame_info


def scatter(ax, X, labels, title):
    for label in np.unique(labels):
        mask = labels == label
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            s=10,
            alpha=0.75,
            color="0.65" if label == -1 else plt.get_cmap("tab20")(int(label) % 20),
            marker="x" if label == -1 else "o",
            label=str(label),
        )
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")


def eda(ds, store, show=True):
    frame = ds.frame
    prefix = f"eda/{ds.name}"
    directory = store.root / prefix
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "info.txt").write_text(frame_info(frame), encoding="utf-8")
    store.table(f"{prefix}/head", frame.head(10))
    store.table(
        f"{prefix}/describe", frame.describe().T.rename_axis("feature").reset_index()
    )
    store.table(
        f"{prefix}/missing",
        frame.isna().sum().rename("missing").rename_axis("feature").reset_index(),
    )
    store.table(
        f"{prefix}/class_counts",
        pd.Series(ds.y)
        .value_counts()
        .sort_index()
        .rename("count")
        .rename_axis("class")
        .reset_index(),
    )
    store.json(
        f"{prefix}/summary.json",
        {
            "rows": len(frame),
            "columns": frame.shape[1],
            "duplicate_feature_rows": int(frame.duplicated().sum()),
            "missing_cells": int(frame.isna().sum().sum()),
            "metadata": ds.meta,
        },
    )
    print(f"\n{ds.name}:\n{frame.head().to_string()}\n{frame_info(frame)}")
    if ds.meta["kind"] == "synthetic":
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
        scatter(axes[0], frame.to_numpy(), ds.y, ds.name)
        for column in frame:
            axes[1].hist(frame[column], bins=25, alpha=0.5, label=column)
        axes[1].legend()
        axes[1].set_title("Coordinate distributions")
        sns.boxplot(data=frame, ax=axes[2])
        axes[2].set_title("Coordinate ranges / outliers")
        fig.tight_layout()
        store.figure(f"eda/{ds.name}/overview", fig, show=show)
    else:
        columns = len(frame.columns)
        ncols = 4
        nrows = int(np.ceil(columns / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.5 * nrows), squeeze=False)
        for ax, column in zip(axes.flat, frame):
            ax.hist(frame[column].dropna(), bins=25)
            ax.set_title(column, fontsize=9)
        for ax in list(axes.flat)[columns:]:
            ax.set_visible(False)
        fig.tight_layout()
        store.figure(f"eda/{ds.name}/histograms", fig, show)
        # Separate axes: do not squash features with different physical units.
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.2 * nrows), squeeze=False)
        for ax, column in zip(axes.flat, frame):
            sns.boxplot(x=frame[column], ax=ax, fliersize=2)
            ax.set_title(column, fontsize=9)
            ax.set_xlabel("")
        for ax in list(axes.flat)[columns:]:
            ax.set_visible(False)
        fig.tight_layout()
        store.figure(f"eda/{ds.name}/boxplots", fig, show)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.heatmap(frame.corr(), cmap="vlag", center=0, ax=axes[0], cbar=False)
        axes[0].tick_params(labelsize=7)
        counts = pd.Series(ds.y).value_counts().sort_index()
        axes[1].bar(counts.index.astype(str), counts.values)
        axes[1].set_title("Reference classes (evaluation only)")
        fig.tight_layout()
        store.figure(f"eda/{ds.name}/correlation_classes", fig, show)


def preprocessing_plot(ds, record, store):
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
    sns.boxplot(data=ds.frame, ax=axes[0], fliersize=1)
    sns.boxplot(
        data=pd.DataFrame(record["X"], columns=record["columns"]),
        ax=axes[1],
        fliersize=1,
    )
    for ax, title in zip(axes, ("Original units", "Prepared features")):
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=90, labelsize=7)
    fig.tight_layout()
    store.figure(f"preprocessing/{record['name']}", fig)


def compare(record, results, store, show=True):
    X = record["X"]
    projection = X if X.shape[1] == 2 else PCA(n_components=2).fit_transform(X)
    fig, axes = plt.subplots(1, 1 + len(results), figsize=(4 * (1 + len(results)), 3.6))
    scatter(axes[0], projection, record["y"], "Reference labels")
    for ax, (algorithm, result) in zip(axes[1:], results.items()):
        scatter(ax, projection, result.labels, algorithm)
    fig.suptitle(record["name"] + (" (PCA display only)" if X.shape[1] != 2 else ""))
    fig.tight_layout()
    store.figure(f"clusters/{record['name']}", fig, show)


def intermediates(record, results, store):
    X = record["X"]
    if X.shape[1] != 2:
        return
    if "PAM" in results:
        result = results["PAM"]
        fig, ax = plt.subplots(figsize=(6, 4))
        scatter(ax, X, result.labels, "PAM medoids")
        medoids = result.arrays["medoids"]
        ax.scatter(*medoids.T, c="black", marker="*", s=180)
        store.figure(f"intermediates/{record['name']}/pam", fig)
    if "GMM" in results:
        result = results["GMM"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        scatter(axes[0], X, result.labels, "GMM components (2 std)")
        covariances = result.arrays["covariances"]
        if covariances.ndim == 2:  # diagonal covariance
            covariances = np.array([np.diag(c) for c in covariances])
        for mean, covariance in zip(result.arrays["means"], covariances):
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            direction = eigenvectors[:, -1]
            angle = np.degrees(np.arctan2(direction[1], direction[0]))
            ellipse = Ellipse(
                mean,
                4 * np.sqrt(eigenvalues[-1]),
                4 * np.sqrt(eigenvalues[0]),
                angle=angle,
                fill=False,
                edgecolor="black",
            )
            axes[0].add_patch(ellipse)
        uncertainty = 1 - result.arrays["probabilities"].max(axis=1)
        image = axes[1].scatter(*X.T, c=uncertainty, s=12, cmap="magma")
        fig.colorbar(image, ax=axes[1])
        axes[1].set_title("1 − maximum posterior probability")
        store.figure(f"intermediates/{record['name']}/gmm", fig)
    if "WaveCluster" in results:
        result = results["WaveCluster"]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for ax, key in zip(axes, ("histogram", "density", "components")):
            ax.imshow(
                result.arrays[key].T, origin="lower", aspect="equal", cmap="viridis"
            )
            ax.set_title(key)
        fig.suptitle("WaveCluster / Haar-DWT variant")
        store.figure(f"intermediates/{record['name']}/wavecluster", fig)
    if "MCL" in results:
        result = results["MCL"]
        graph = result.matrices["graph"].tocoo()
        mask = graph.row < graph.col
        edges = np.column_stack([graph.row[mask], graph.col[mask]])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].add_collection(LineCollection(X[edges], linewidths=0.3, colors="0.7"))
        scatter(axes[0], X, result.labels, "MCL weighted kNN graph")
        history = result.arrays["history"]
        axes[1].semilogy(history[:, 0], np.maximum(history[:, 1], 1e-16))
        axes[1].set(
            xlabel="Iteration", ylabel="Maximum matrix change", title="MCL convergence"
        )
        store.figure(f"intermediates/{record['name']}/mcl", fig)


def metric_figures(frame, store):
    valid = frame[frame.status == "ok"].copy()
    if valid.empty:
        return
    for metric in ("ari_all", "silhouette", "coverage", "n_clusters"):
        pivot = valid.pivot(index="dataset", columns="algorithm", values=metric)
        fig, ax = plt.subplots(figsize=(9, max(4, len(pivot) * 0.24)))
        sns.heatmap(pivot, annot=len(pivot) <= 12, fmt=".2f", cmap="viridis", ax=ax)
        ax.set_title(metric)
        fig.tight_layout()
        store.figure(f"summary/{metric}", fig)


def tuning_figures(frame, store):
    valid = frame[frame.status == "ok"].copy()
    if valid.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(data=valid, x="algorithm", y="silhouette", ax=axes[0])
    sns.scatterplot(
        data=valid, x="n_clusters", y="silhouette", hue="algorithm", ax=axes[1]
    )
    axes[0].set_title("Parameter sensitivity: internal silhouette")
    axes[1].set_title("Cluster count vs internal criterion (not ground-truth quality)")
    fig.tight_layout()
    store.figure("summary/parameter_sensitivity", fig)
    for name in ("ellipses_rotated_deterministic", "rings_PCG64_42", "wine_pca2"):
        subset = valid[valid.dataset == name]
        if subset.empty:
            continue
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
        for ax, algorithm, key in zip(
            axes,
            ("PAM", "WaveCluster", "GMM", "MCL"),
            ("k", "significance", "k", "inflation"),
        ):
            group = subset[subset.algorithm == algorithm].copy()
            group["parameter"] = [json.loads(p)[key] for p in group.params]
            metric = "bic" if algorithm == "GMM" else "silhouette"
            sns.lineplot(
                data=group, x="parameter", y=metric, marker="o", errorbar=None, ax=ax
            )
            ax.set(xlabel=key, ylabel=metric, title=algorithm)
        fig.suptitle(name + " (mean over other candidate settings)")
        fig.tight_layout()
        store.figure(f"summary/parameters_{name}", fig)


def stability_figures(frame, store):
    valid = frame[frame.status == "ok"]
    if valid.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for algorithm, group in valid.groupby("algorithm"):
        mean = group.groupby("delta_relative")[["ari", "changed_fraction"]].mean()
        axes[0].plot(mean.index, mean.ari, "o-", label=algorithm)
        axes[1].plot(mean.index, mean.changed_fraction, "o-", label=algorithm)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("δ / characteristic distance")
        ax.legend()
    axes[0].set_ylabel("ARI to baseline (mean)")
    axes[1].set_ylabel("Aligned changed fraction (mean)")
    fig.tight_layout()
    store.figure("stability/overview", fig)


def benchmark_figures(summary, store):
    valid = summary[summary.status == "ok"]
    if valid.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for (algorithm, source), group in valid.groupby(["algorithm", "source"]):
        for ax, metric in zip(
            axes, ("elapsed_s_median", "peak_rss_mb_median", "increment_mb_median")
        ):
            ax.plot(group.n, group[metric], "o-", label=f"{algorithm}/{source}")
            ax.set(xlabel="n", ylabel=metric)
            ax.set_xscale("log")
    axes[-1].legend(fontsize=7, bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    store.figure("benchmarks/scaling", fig)
