from __future__ import annotations

import unittest
from pathlib import Path

from data_agent_baseline.semantic.coverage import render_coverage_report, scan_semantic_coverage


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SemanticCoverageTests(unittest.TestCase):
    def test_scanner_reports_current_models_and_task_287_rejection(self) -> None:
        report = scan_semantic_coverage(
            PROJECT_ROOT / "public" / "input",
            PROJECT_ROOT / "semantic_models",
        )
        self.assertEqual(report["summary"]["task_count"], 50)
        self.assertEqual(report["summary"]["model_count"], 2)

        tasks = {item["task_id"]: item for item in report["tasks"]}
        self.assertEqual(tasks["task_169"]["matched_models"], ["debit_card"])
        self.assertEqual(tasks["task_269"]["matched_models"], ["superhero"])
        self.assertEqual(tasks["task_287"]["matched_models"], [])
        superhero_decision = next(
            item for item in tasks["task_287"]["decisions"] if item["model_id"] == "superhero"
        )
        self.assertTrue(superhero_decision["source_compatible"])
        self.assertFalse(superhero_decision["matched"])
        self.assertIn("gender", superhero_decision["unmapped_source_fields"])
        self.assertTrue(
            any("not exposed by model" in reason for reason in superhero_decision["reasons"])
        )

        rendered = render_coverage_report(report)
        self.assertIn("# Semantic Coverage Report", rendered)
        self.assertIn("task_287", rendered)
        self.assertIn("REJECT superhero", rendered)


if __name__ == "__main__":
    unittest.main()
