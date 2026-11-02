"""One isolated fit; invoked only by src.benchmark's subprocess controller."""

import json
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np
import psutil

from .algorithms import run_algorithm
from .artifacts import write_json


def main():
    request = Path(sys.argv[1])
    config = json.loads(request.read_text())
    directory = request.parent
    X = np.load(config["input"])["X"]
    process = psutil.Process()
    # Import and input loading finish before the fit timer / RSS sampler start.
    write_json(
        directory / "ready.json", {"baseline_rss_mb": process.memory_info().rss / 2**20}
    )
    deadline = time.monotonic() + 120
    while not (directory / "go").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Parent did not start measurement")
        time.sleep(0.005)
    try:
        start = time.perf_counter()
        result = run_algorithm(config["algorithm"], X, config["params"], config["seed"])
        elapsed = time.perf_counter() - start
        # Wait for the parent to stop its RSS sampler before serializing labels.
        write_json(directory / "fit_finished.json", {"elapsed_s": elapsed})
        deadline = time.monotonic() + 30
        while (
            config.get("acknowledge_fit", False)
            and not (directory / "measurement_done").exists()
        ):
            if time.monotonic() > deadline:
                raise TimeoutError("Parent did not acknowledge fit completion")
            time.sleep(0.005)
        np.savez_compressed(directory / "labels.npz", labels=result.labels)
        write_json(
            directory / "result.json",
            {
                "status": "ok",
                "elapsed_s": elapsed,
                "lifetime_highwater_mb": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                / 1024,
                "n_clusters": len(set(result.labels) - {-1}),
                "diagnostics": result.diagnostics,
            },
        )
    except Exception as error:
        write_json(
            directory / "result.json",
            {
                "status": "error",
                "reason": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )


if __name__ == "__main__":
    main()
