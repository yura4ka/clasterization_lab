"""Bounded subprocess benchmarking; no notebook/kernel memory is attributed to fits."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil

from .algorithms import ALGORITHMS, default_params
from .artifacts import write_json
from .data import geometry


def isolated_fit(
    X, algorithm, params, seed, directory, project, timeout=60, memory_budget_mb=1024
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "input.npz", X=X)
    request = directory / "request.json"
    write_json(
        request,
        {
            "input": str(directory / "input.npz"),
            "algorithm": algorithm,
            "params": params,
            "seed": seed,
            "acknowledge_fit": True,
        },
    )
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    with (directory / "stdout.log").open("w") as stdout, (
        directory / "stderr.log"
    ).open("w") as stderr:
        child = subprocess.Popen(
            [sys.executable, "-m", "src.benchmark_worker", str(request)],
            cwd=project,
            env=env,
            stdout=stdout,
            stderr=stderr,
        )
        process = psutil.Process(child.pid)
        try:
            start_import = time.monotonic()
            while not (directory / "ready.json").exists():
                if child.poll() is not None:
                    return {
                        "status": "error",
                        "reason": "Worker exited before ready; see stderr.log",
                    }
                if time.monotonic() - start_import > 90:
                    child.kill()
                    child.wait()
                    return {
                        "status": "timeout",
                        "reason": "Import/input readiness exceeded 90 s",
                    }
                time.sleep(0.01)
            baseline = json.loads((directory / "ready.json").read_text())[
                "baseline_rss_mb"
            ]
            peak, started = baseline, time.monotonic()
            (directory / "go").touch()
            status = None
            while child.poll() is None:
                try:
                    peak = max(peak, process.memory_info().rss / 2**20)
                except psutil.NoSuchProcess:
                    break
                if time.monotonic() - started > timeout:
                    status = "timeout"
                elif peak - baseline > memory_budget_mb:
                    status = "memory_limit"
                if status:
                    child.kill()
                    break
                if (directory / "fit_finished.json").exists():
                    (directory / "measurement_done").touch()
                    break
                if (directory / "result.json").exists():
                    break
                time.sleep(0.005)
            child.wait(timeout=15)
            output = {
                "baseline_rss_mb": baseline,
                "peak_rss_mb": peak,
                "increment_mb": max(0.0, peak - baseline),
                "sampling_interval_ms": 5,
            }
            if status:
                return {
                    **output,
                    "status": status,
                    "reason": "Configured fit resource limit",
                }
            result_path = directory / "result.json"
            if result_path.exists():
                return {**output, **json.loads(result_path.read_text())}
            return {
                **output,
                "status": "error",
                "reason": f"Worker exit code {child.returncode}",
            }
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()


def run_benchmarks(real_records, store, mode, seed=42):
    sizes = [120, 300] if mode == "quick" else [150, 300, 600, 1200]
    repetitions = 2 if mode == "quick" else 3
    rows = []
    sources = ["synthetic"] if mode == "quick" else ["synthetic", "dry_bean_pca2"]
    for source in sources:
        for n in sizes:
            if source == "synthetic":
                ds = geometry("ellipses_rotated", int(np.ceil(n / 3)), "PCG64", seed)
                X = ds.frame.to_numpy()[:n]
                X = (X - X.mean(axis=0)) / max(float(X.std()), 1e-12)
            else:
                # Use a nested, fixed-random-order subset of the full processed source.
                X_all = real_records[source]["X"]
                if n > len(X_all):
                    continue
                order = np.random.default_rng(seed).permutation(len(X_all))
                X = X_all[order[:n]]
            for algorithm in ALGORITHMS:
                params = default_params(algorithm, X)
                for repetition in range(repetitions):
                    key = f"{source}/{algorithm}/n{n}_r{repetition}"
                    result = isolated_fit(
                        X,
                        algorithm,
                        params,
                        seed,
                        store.root / "benchmarks" / "raw" / key,
                        store.project,
                    )
                    result.pop("diagnostics", None)
                    rows.append(
                        {
                            "source": source,
                            "algorithm": algorithm,
                            "n": n,
                            "d": X.shape[1],
                            "repetition": repetition,
                            **result,
                        }
                    )
                    store.table("benchmarks/raw_results", pd.DataFrame(rows))
            store.logger.info("Benchmark completed: %s n=%d", source, n)
    raw = pd.DataFrame(rows)
    numeric = ["elapsed_s", "peak_rss_mb", "baseline_rss_mb", "increment_mb"]
    summary_rows = []
    for keys, group in raw.groupby(["source", "algorithm", "n"]):
        successful = group[group.status == "ok"]
        row = dict(zip(["source", "algorithm", "n"], keys))
        row.update(
            status="ok" if len(successful) else "failed",
            successful_repeats=len(successful),
            total_repeats=len(group),
        )
        for metric in numeric:
            values = (
                successful[metric] if metric in successful else pd.Series(dtype=float)
            )
            row[f"{metric}_median"] = values.median()
            row[f"{metric}_q25"] = values.quantile(0.25)
            row[f"{metric}_q75"] = values.quantile(0.75)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    store.table("benchmarks/summary", summary)
    store.json(
        "benchmarks/protocol.json",
        {
            "repetitions": repetitions,
            "sizes": sizes,
            "fit_timeout_s": 60,
            "additional_rss_limit_mb": 1024,
            "threads": 1,
            "timer": "fit including distance matrix / graph / grid; excludes import and file I/O",
            "rss": "5 ms sampled worker RSS, baseline after imports/input load; may miss short peaks",
            "rss_sampling_tail": "worker waits for sampler acknowledgment before label serialization",
            "lifetime_highwater": "separate OS highwater includes import phase, not incremental fit peak",
            "configuration": "fixed default parameters, k=3, not selected by benchmark results",
            "warmup": "fresh process per fit; imports completed, no discarded fit warmup",
        },
    )
    return raw, summary
