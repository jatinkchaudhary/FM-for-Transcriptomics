import unittest

import numpy as np
import torch
import torch.nn as nn

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


if __name__ == "__main__":
    unittest.main()
