import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.backend.atlas_runtime import AtlasRuntime, AtlasUnavailableError


class AtlasRuntimeTest(unittest.TestCase):
    def test_matches_species_and_tissue_from_reference_expression(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez_compressed(
                root / "atlas.npz",
                genes=np.asarray(["A", "B", "C"]),
                reference_ids=np.asarray(["human_liver", "mouse_brain"]),
                expression=np.asarray([[4.0, 1.0, 0.0], [0.0, 1.0, 4.0]], dtype=np.float32),
            )
            with (root / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["reference_id", "species", "tissue", "source"]
                )
                writer.writeheader()
                writer.writerow(
                    {"reference_id": "human_liver", "species": "human", "tissue": "liver", "source": "test"}
                )
                writer.writerow(
                    {"reference_id": "mouse_brain", "species": "mouse", "tissue": "brain", "source": "test"}
                )
            (root / "diseases.json").write_text(
                json.dumps({"Example association": ["A", "B"]}), encoding="utf-8"
            )
            runtime = AtlasRuntime(
                {
                    "index": str(root / "atlas.npz"),
                    "metadata": str(root / "metadata.csv"),
                    "disease_gene_sets": str(root / "diseases.json"),
                    "minimum_shared_genes": 3,
                    "ollama": {"enabled": False, "model": "test"},
                }
            )
            # Runtime input is raw-scale and is transformed with log1p before matching.
            expression = np.expm1(np.asarray([[4.0], [1.0], [0.0]], dtype=np.float32))
            result = runtime.analyze(["A", "B", "C"], ["S1"], expression)

            best = result["sample_results"][0]["matches"][0]
            self.assertEqual(best["reference_id"], "human_liver")
            self.assertEqual(best["species"], "human")
            self.assertEqual(result["language_head"]["status"], "disabled")
            self.assertEqual(result["input"]["shared_genes"], 3)

    def test_unconfigured_atlas_fails_explicitly(self):
        with self.assertRaisesRegex(AtlasUnavailableError, "No local atlas index"):
            AtlasRuntime({}).analyze(["A"], ["S1"], np.ones((1, 1), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
