import tempfile
import unittest

import numpy as np
from scipy import sparse
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from src.algorithms import ALGORITHMS, default_params, run_algorithm, normalize
from src.artifacts import Store
from src.data import geometry, prepare, pca_record
from src.experiments import aligned_change, quality, tune


class AlgorithmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = make_blobs(
            n_samples=90,
            centers=[[-4, -3], [0, 4], [4, -3]],
            cluster_std=0.2,
            random_state=42,
        )

    def test_control_clusters_and_reproducibility(self):
        for algorithm in ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                params = default_params(algorithm, self.X)
                if algorithm == "MCL":
                    # A control configuration, not a claim that every inflation
                    # recovers a prescribed cluster count.
                    params["inflation"] = 1.5
                first = run_algorithm(algorithm, self.X, params)
                second = run_algorithm(algorithm, self.X, params)
                np.testing.assert_array_equal(first.labels, second.labels)
                self.assertGreater(adjusted_rand_score(self.y, first.labels), 0.9)
                if algorithm == "MCL":
                    np.testing.assert_allclose(
                        np.asarray(first.matrices["flow"].sum(axis=0)), 1
                    )
                if algorithm == "GMM":
                    np.testing.assert_allclose(
                        first.arrays["probabilities"].sum(axis=1), 1
                    )

    def test_label_alignment_and_noise(self):
        self.assertEqual(aligned_change([0, 0, 1, -1], [5, 5, 4, -1]), 0)
        self.assertEqual(aligned_change([0, 0, 1, -1], [5, 5, 4, 4]), 0.25)
        self.assertEqual(aligned_change([-1, -1], [0, 0]), 1)

    def test_invalid_internal_scores(self):
        scores = quality(self.X, self.y, np.full(len(self.X), -1))
        self.assertTrue(np.isnan(scores["silhouette"]))
        self.assertEqual(scores["coverage"], 0)
        self.assertTrue(scores["metric_note"])

    def test_wavecluster_fixed_bounds(self):
        params = default_params("WaveCluster", self.X)
        changed = self.X.copy()
        changed[0] = [100, 100]
        result = run_algorithm("WaveCluster", changed, params)
        self.assertEqual(result.labels[0], -1)
        self.assertEqual(result.diagnostics["out_of_bounds"], 1)
        with self.assertRaises(ValueError):
            run_algorithm("WaveCluster", np.ones((10, 3)))

    def test_haar_density_mapping(self):
        result = run_algorithm("WaveCluster", self.X)
        hist = result.arrays["histogram"]
        expected = hist.reshape(16, 2, 16, 2).sum(axis=(1, 3)) / 4
        np.testing.assert_allclose(result.arrays["density"], expected, atol=1e-12)

    def test_stochastic_versus_deterministic(self):
        a = geometry("circles", 30, "PCG64")
        b = geometry("circles", 30, "MT19937")
        self.assertFalse(np.array_equal(a.frame, b.frame))

    def test_invalid_algorithm_parameters(self):
        with self.assertRaises(ValueError):
            run_algorithm("PAM", self.X, {"k": 0})
        with self.assertRaises(ValueError):
            run_algorithm("MCL", self.X, {"inflation": 1.0})

    def test_degenerate_duplicates(self):
        X = np.zeros((12, 2))
        for algorithm in ("WaveCluster", "MCL", "PAM"):
            self.assertEqual(len(run_algorithm(algorithm, X).labels), len(X))

    def test_column_normalization(self):
        M = normalize(sparse.csc_matrix([[2.0, 1.0], [1.0, 2.0]]))
        np.testing.assert_allclose(np.asarray(M.sum(axis=0)), 1)

    def test_generators_and_saved_preparation(self):
        for generator in ("deterministic", "PCG64", "MT19937"):
            a = geometry("mixture", 30, generator, noise=5)
            b = geometry("mixture", 30, generator, noise=5)
            np.testing.assert_array_equal(a.frame, b.frame)
            self.assertEqual(np.count_nonzero(a.y == -1), 5)
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory, "test")
            record = prepare(a, 40, 42, store)
            self.assertEqual(record["X"].shape, (40, 2))
            self.assertTrue(np.isfinite(record["X"]).all())
            projected = pca_record(record, store)
            self.assertEqual(projected["X"].shape, (40, 2))
            store.finish()

    def test_selection_does_not_use_reference_labels(self):
        X, y = self.X[:45], self.y[:45]
        record = {"name": "control", "X": X, "y": y}
        with tempfile.TemporaryDirectory() as directory:
            first, _ = tune({"control": record}, Store(directory, "first"), "quick")
            shuffled = {**record, "y": np.random.default_rng(7).permutation(y)}
            second, _ = tune({"control": shuffled}, Store(directory, "second"), "quick")
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
