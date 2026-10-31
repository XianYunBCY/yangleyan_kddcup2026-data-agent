# PHASE 1 Demo Data

## Dataset Overview

PHASE 1 Demo Data is a task-based data analysis benchmark dataset for evaluating model performance on structured data, databases, and knowledge document scenarios. The dataset contains two main directories, **input** and **output**, both organized by task. Public demo data includes both directories; hidden test sets only include `input/`.

Each task's **input** includes the task id, difficulty level, question, and task-specific context data. Each task's **output** directory provides the ground truth `gold.csv` for evaluation and comparison.

## Directory Structure

```text
data/public/
├── input/                 # Input data, organized by task
│   └── task_<id>/
│       ├── task.json      # Task metadata (task_id, difficulty, question)
│       └── context/       # Task-specific context data
│           ├── csv/       # Optional structured CSV files
│           ├── db/        # Optional database files (SQLite)
│           ├── json/      # Optional structured JSON files
│           ├── doc/       # Optional data documents (Markdown)
│           └── knowledge.md
└── output/                # Output data, organized by task
    └── task_<id>/
        └── gold.csv       # Ground truth
```

Notes:
- The exact contents under `context/` are not fixed across tasks.
- Each task includes at least one context source such as `csv/`, `db/`, `json/`, or `doc/`.
- In this demo package, each task under `output/` contains `gold.csv`.

## Input Format

Each task's input contains the following fields:

| Field | Description |
|-------|-------------|
| **task_id** | Unique task identifier |
| **difficulty** | Difficulty level (`easy`, `medium`, `hard`, or `extreme`), see Difficulty Levels table below |
| **question** | The question to be analyzed |
| **context** | Task-specific context data under `context/`, including one or more sources such as `csv/`, `db/`, `json/`, `doc/`, and `knowledge.md` |

## Output Format

- **gold.csv**: Ground truth for evaluation and comparison.

## PHASE 1 DIFFICULTY LEVELS

| Level | Modalities | Core Challenge | Document Scale |
|-------|------------|----------------|----------------|
| Easy | Structured files such as CSV/JSON plus knowledge documents | Code generation for data analysis workflows such as Python execution | Short context |
| Medium | Structured files, databases, and knowledge documents | Text-to-SQL and multi-source data analysis | Moderate context |
| Hard | Structured files, databases, data documents, and knowledge documents | Reasoning over unstructured data documents | ~10K-128K tokens |
| Extreme | Same modality combination as Hard | Context engineering and memory under ultra-long document inputs | >128K tokens |
