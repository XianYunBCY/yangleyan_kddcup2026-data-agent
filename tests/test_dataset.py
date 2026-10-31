from __future__ import annotations

import json

from data_agent_baseline.benchmark.dataset import DABenchPublicDataset


def test_dataspace_task_without_difficulty_is_supported(tmp_path) -> None:
    task_dir = tmp_path / "task_87"
    (task_dir / "context").mkdir(parents=True)
    (task_dir / "task.json").write_text(
        json.dumps({"task_id": "task_87", "question": "What happened?"})
    )

    task = DABenchPublicDataset(tmp_path).get_task("task_87")

    assert task.task_id == "task_87"
    assert task.question == "What happened?"
    assert task.difficulty == "unknown"


def test_legacy_task_difficulty_is_preserved(tmp_path) -> None:
    task_dir = tmp_path / "task_1"
    (task_dir / "context").mkdir(parents=True)
    (task_dir / "task.json").write_text(
        json.dumps(
            {
                "task_id": "task_1",
                "difficulty": "easy",
                "question": "What happened?",
            }
        )
    )

    task = DABenchPublicDataset(tmp_path).get_task("task_1")

    assert task.difficulty == "easy"
