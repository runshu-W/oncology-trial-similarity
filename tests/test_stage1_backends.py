from __future__ import annotations

import importlib
import unittest


pipeline = importlib.import_module("docs.oncology_trial_similarity_pipeline")


class Stage1BackendTests(unittest.TestCase):
    def test_default_retrieval_backend_is_clinicalbert_compatible(self) -> None:
        self.assertEqual(pipeline.DEFAULT_RETRIEVAL_BACKEND, "clinicalbert")
        self.assertIn("clinicalbert", pipeline.RETRIEVAL_BACKENDS)
        self.assertIn("trial2vec", pipeline.RETRIEVAL_BACKENDS)
        self.assertIn("secret", pipeline.RETRIEVAL_BACKENDS)

    def test_secret_backend_is_explicitly_not_implemented(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "SECRET retrieval"):
            pipeline.ensure_supported_retrieval_backend("secret")

    def test_unknown_backend_has_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported retrieval backend"):
            pipeline.ensure_supported_retrieval_backend("not_a_backend")


if __name__ == "__main__":
    unittest.main()
