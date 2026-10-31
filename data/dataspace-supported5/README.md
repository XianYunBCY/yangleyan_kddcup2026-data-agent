# DataSpace strict supported public-reference subset

This five-task subset is selected from the 60 official public-reference tasks.
Every selected task:

- has an official `gold.csv` and frozen evaluation config;
- contains only formats supported by the current runner: CSV, JSON, SQLite,
  Markdown, and plain text;
- contains no PDF or video files.

The task directories, gold directories, and configs are symbolic links to the
locally extracted official DataSpace release, so no benchmark data is copied.

| Task | Workspace formats | Context size | Focus |
| --- | --- | ---: | --- |
| `task_44` | CSV, SQLite, Markdown | 91.7 MB | latest timestamp, ordered multi-row result |
| `task_87` | CSV, SQLite, Markdown | 74.0 MB | temporal comparison |
| `task_127` | CSV, JSON, SQLite | 343.5 MB | filtered numeric maximum |
| `task_329` | CSV, JSON, SQLite, Markdown | 115.4 MB | encounter-scoped daily aggregation |
| `task_409` | CSV, SQLite, Markdown | 102.0 MB | date difference and first-stay selection |

Run the subset:

```bash
uv run dabench run-benchmark \
  --config configs/react_baseline.dataspace-supported5.yaml
```

The config reads the model credential from `MODEL_API_KEY` and does not store a
secret in the repository.

Evaluate a completed run with the official evaluator, using this directory as
the five-task config root:

```bash
.venv/bin/python official-dataspace/evaluation/evaluate.py \
  --prediction-root artifacts/runs/<run_id> \
  --gold-root data/dataspace-supported5/output \
  --config-root data/dataspace-supported5/evaluation/configs \
  --output artifacts/runs/<run_id>/official_evaluation_summary.json
```

The resulting Task Accuracy denominator is 5. This subset is a strict harness
compatibility test, not a representative estimate of full six-modality
DataSpace accuracy: all five tasks are English-language healthcare tasks.
