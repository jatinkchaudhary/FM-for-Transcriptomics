import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

from app.backend.downstream_analysis import build_live_analysis
from app.backend.model_runtime import (
    INFERENCE_BATCH_SAMPLES,
    MAX_REQUEST_SAMPLES,
    ModelRuntime,
    RequestError,
)


class RecordingDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, values):
        self.batch_sizes.append(len(values))
        return values + 1


class ModelRuntimeTest(unittest.TestCase):
    def test_validate_payload_accepts_builtin_example_sample_count(self):
        samples = [f"sample_{index}" for index in range(46)]
        payload = {
            "genes": ["TP53"],
            "samples": samples,
            "matrix": [[None] + [1.0] * 45],
            "missing": [[True] + [False] * 45],
        }

        genes, parsed_samples, values, missing = ModelRuntime._validate_payload(payload)

        self.assertEqual(genes, ["TP53"])
        self.assertEqual(parsed_samples, samples)
        self.assertEqual(values.shape, (1, 46))
        self.assertTrue(missing[0, 0])

    def test_validate_payload_rejects_request_above_sample_limit(self):
        sample_count = MAX_REQUEST_SAMPLES + 1
        payload = {
            "genes": ["TP53"],
            "samples": [f"sample_{index}" for index in range(sample_count)],
            "matrix": [[None] * sample_count],
            "missing": [[True] * sample_count],
        }

        with self.assertRaisesRegex(
            RequestError, rf"{MAX_REQUEST_SAMPLES:,} samples"
        ):
            ModelRuntime._validate_payload(payload)

    def test_prediction_is_split_into_safe_batches(self):
        runtime = ModelRuntime.__new__(ModelRuntime)
        runtime.device = torch.device("cpu")
        runtime.model = RecordingDecoder()
        aligned = np.zeros((46, 5), dtype=np.float32)

        prediction = runtime._predict_aligned("Txn_Jatin", aligned)

        self.assertEqual(
            runtime.model.batch_sizes,
            [INFERENCE_BATCH_SAMPLES, 46 - INFERENCE_BATCH_SAMPLES],
        )
        np.testing.assert_array_equal(prediction, np.ones_like(aligned))

    def test_live_analysis_is_derived_from_submitted_embeddings(self):
        common = {
            "model": "Txn_Jatin",
            "embedding_mode": "test embedding",
            "genes": ["A", "B", "C"],
            "samples": ["S1", "S2", "S3", "S4"],
            "groups": ["case", "case", "control", "control"],
            "expression": np.asarray(
                [[5, 4, 1, 0], [0, 1, 4, 5], [1, 1, 1, 1]],
                dtype=np.float32,
            ),
            "matched_genes": 3,
            "model_gene_count": 3,
            "methods": ("pca",),
        }
        first = build_live_analysis(
            embeddings=np.asarray(
                [[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]],
                dtype=np.float32,
            ),
            **common,
        )
        second = build_live_analysis(
            embeddings=np.asarray(
                [[1, 0], [0, 1], [1, 0], [0, 1]],
                dtype=np.float32,
            ),
            **common,
        )

        self.assertEqual(first["source"], "current_request")
        self.assertEqual(first["scope"], "uploaded cohort")
        self.assertIn("pca", first["projections"])
        self.assertNotEqual(first["similarity"], second["similarity"])
        self.assertEqual(first["nearest_neighbors"][0]["neighbors"][0]["sample"], "S2")

    def test_expression_fallback_accepts_complete_matrix(self):
        runtime = ModelRuntime.__new__(ModelRuntime)
        runtime.public_models = {
            "ESM3": {
                "id": "ESM3",
                "label": "ESM3",
                "imputation_supported": False,
            }
        }
        payload = {
            "model": "ESM3",
            "genes": ["A", "B"],
            "samples": ["S1", "S2"],
            "groups": ["case", "control"],
            "matrix": [[1.0, 3.0], [4.0, 2.0]],
            "missing": [[False, False], [False, False]],
            "input_scale": "log1p",
        }

        with patch(
            "app.backend.model_runtime.build_live_analysis",
            return_value={"source": "current_request"},
        ) as builder:
            result = runtime.analyze_downstream(payload)

        self.assertEqual(result["source"], "current_request")
        self.assertEqual(result["device"], "CPU")
        arguments = builder.call_args.kwargs
        self.assertEqual(arguments["embedding_mode"], "standardized log-expression profile")
        self.assertEqual(arguments["embeddings"].shape, (2, 2))
        self.assertFalse(np.allclose(arguments["embeddings"][0], arguments["embeddings"][1]))


if __name__ == "__main__":
    unittest.main()
