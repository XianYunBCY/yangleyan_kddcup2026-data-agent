from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from data_agent_baseline.config import load_app_config


class HarnessConfigTests(unittest.TestCase):
    def test_loads_semantic_mode_and_versions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(
                """
harness:
  mode: semantic
  prompt_version: semantic-prompt-v1
  semantic_model_version: catalog-2026-08
""".strip()
            )
            config = load_app_config(config_path)
            self.assertEqual(config.harness.mode, "semantic")
            self.assertEqual(config.harness.prompt_version, "semantic-prompt-v1")
            self.assertEqual(config.harness.semantic_model_version, "catalog-2026-08")

    def test_rejects_unknown_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text("harness:\n  mode: unknown\n")
            with self.assertRaises(ValueError):
                load_app_config(config_path)


if __name__ == "__main__":
    unittest.main()
