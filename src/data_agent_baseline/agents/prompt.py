from __future__ import annotations

import csv
import json
import re
import sqlite3

from data_agent_baseline.benchmark.schema import PublicTask


REACT_SYSTEM_PROMPT = """
You are a ReAct-style data agent.

You are solving a data task. You may only inspect files inside the task's `context/` directory through the provided tools.

Rules:
1. Use tools to inspect the available context before answering.
2. Base your answer only on information you can observe through the provided tools.
3. The task is complete only when you call a terminal answer tool: `answer`,
   `answer_duckdb`, `answer_context_sql`, or `answer_prepared_asq` when available.
4. The `answer` tool must receive a table with `columns` and `rows`.
5. Always return exactly one JSON object with keys `thought`, `action`, and `action_input`.
6. Always wrap that JSON object in exactly one fenced code block that starts with ```json and ends with ```.
7. Do not output any text before or after the fenced JSON block.
8. When passing Python code, keep it valid JSON: escape newlines as `\n` inside the `code`
   string, or use a compact one-line program.

CRITICAL - Computation and answer table rules (violations will cause wrong answers):
- Prefer SQL tools for tabular work. Use `execute_context_sql` for SQLite databases and
  `execute_duckdb` for CSV/JSON files.
- When the final result can be produced by one SQL query, call `answer_duckdb` for CSV/JSON
  files or `answer_context_sql` for SQLite/database files. Do not waste a step by calling
  `execute_duckdb` and then copying the result into `answer`.
- Discover the schema from the available files before choosing tables, fields, joins, or
  filters. Do not assume a field exists just because the question uses that wording.
- Treat the question as the contract for the final answer. Submit only the value(s) directly
  asked for; do not add ids, scores, dates, descriptions, ranking metrics, helper columns, or
  intermediate calculations unless the question explicitly asks for them.
- Field choice rule: if an observed field name matches the question wording exactly or after
  simple normalization such as case, spaces, underscores, or pluralization, use that field
  first. Only when no matching field exists should you choose the closest observed field(s) by
  meaning. If a requested concept is represented by multiple observed fields, use those observed
  fields rather than inventing a new merged field.
- For highest/lowest/minimum/maximum questions, use the compared field to filter or order the
  rows, but include that compared field in the final answer only if the question asks for it.
  Aggregate only when the question asks for total, average, count, per-group, or similar
  derived values. Do not treat missing related records as zero unless the data or question says
  so.
- If a question word exactly matches an observed field in a detail table, rank/filter the
  individual detail rows by that exact field first, then join outward only to display the
  requested entity. Do not aggregate that field up to the answer entity unless the question
  explicitly says total, sum, average, count, overall, per-entity, or per-group.
- If a highest/lowest/minimum/maximum comparison has ties at the best value, return every tied
  requested answer row unless the question explicitly asks for one row, top 1, first, latest, or
  a tie-breaker.
- Missing related records are unknown or absent records, not numeric zero. Use zero only when a
  source field actually contains 0 or when the question/data definition explicitly says missing
  means zero.
- Do not invent output columns by merging, renaming, or explaining source fields. If the answer
  concept can be represented by observed fields such as first_name plus last_name, submit those
  observed fields. If there is no observed field for the exact concept, use the closest observed
  field name rather than a new friendly label.
- For long markdown/JSON/CSV documents, use `search_context` to locate relevant entity names,
  IDs, field names, or values before reading large chunks with `read_doc`.
- Never read a whole large document into the conversation. If a text file is large, either:
  use `search_context`, or use `execute_python` to parse the file and print only the final
  small result table or a short diagnostic summary.

CRITICAL - Verification and decisiveness rules:
- Avoid looping on the same approach more than twice. If a method fails, try a different one.
- If the step limit is near, submit the best supported table you have.
""".strip()


SEMANTIC_HARNESS_SYSTEM_PROMPT = (
    REACT_SYSTEM_PROMPT
    + """

CRITICAL - Semantic harness protocol:
- Your first action must be `get_semantic_catalog`. Do this before `list_context`, schema/file
  inspection, SQL, or Python. This overrides the general instruction to inspect raw context first.
- If a deterministic model is returned, use its governed dimensions, metrics, filters, and output
  labels directly; do not inspect the underlying files because the compiler owns their physical
  structure. Only fall back to raw context inspection if no deterministic model matches.
- When a deterministic semantic model is available, call `prepare_asq` with only business-level
  ASQ. Never provide physical paths, columns, joins, or SQL; those are compiler-owned.
- After compilation succeeds, call terminal `answer_prepared_asq` with the returned fingerprint.
- Raw SQL, manual answer submission, and arbitrary Python are unavailable when a deterministic
  semantic compiler matches the task.
- For tasks without a matched deterministic model, the legacy registered-plan protocol may still
  be available as a fallback.
"""
).strip()

RESPONSE_EXAMPLES = """
Example response when you need to inspect the context:
```json
{"thought":"I should inspect the available files first.","action":"list_context","action_input":{"max_depth":4}}
```

Example response when you have the final answer after an SQL query:
```json
{"thought":"I have the final result from the observed data.","action":"answer","action_input":{"columns":["AVG(T1.value)"],"rows":[["63.5"]]}}
```

Example response when the answer has multiple columns from the original data:
```json
{"thought":"I have the final result from the observed data.","action":"answer","action_input":{"columns":["source_field_a","source_field_b"],"rows":[["A","B"],["C","D"]]}}
```
""".strip()


def _build_context_hint(task: PublicTask) -> str:
    entries: list[str] = []
    try:
        files = [path for path in task.context_dir.rglob("*") if path.is_file()]
    except OSError:
        return ""

    for path in sorted(files):
        try:
            size = path.stat().st_size
        except OSError:
            continue
        rel_path = path.relative_to(task.context_dir).as_posix()
        if size > 50000 and path.suffix.lower() in {".md", ".txt", ".json", ".csv", ".tsv"}:
            entries.append(
                f"- {rel_path} ({size} bytes): use `search_context` or aggregate-only Python."
            )
    if not entries:
        return ""
    return "Large-file strategy hints:\n" + "\n".join(entries) + "\n"


def _normalize_schema_name(value: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", value.lower())
    normalized_tokens: list[str] = []
    for token in tokens:
        if len(token) > 3 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        normalized_tokens.append(token)
    return "_".join(normalized_tokens)


def _question_schema_terms(question: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "by",
        "for",
        "from",
        "has",
        "have",
        "had",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "were",
        "what",
        "which",
        "who",
        "with",
    }
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", question.lower())
        if token not in stopwords and len(token) > 1
    ]
    terms = {_normalize_schema_name(token) for token in tokens}
    for index in range(len(tokens) - 1):
        terms.add(_normalize_schema_name(f"{tokens[index]}_{tokens[index + 1]}"))
    return {term for term in terms if term}


def _json_record_fields(payload: object) -> tuple[str | None, list[str]]:
    if isinstance(payload, dict):
        table_name = payload.get("table")
        table = table_name if isinstance(table_name, str) else None
        records = payload.get("records")
        if isinstance(records, list):
            for record in records:
                if isinstance(record, dict):
                    return table, [str(key) for key in record.keys()]
            return table, []
        return table, [str(key) for key in payload.keys()]
    if isinstance(payload, list):
        for record in payload:
            if isinstance(record, dict):
                return None, [str(key) for key in record.keys()]
    return None, []


def _scan_schema_fields(task: PublicTask) -> list[tuple[str, str, str]]:
    fields: list[tuple[str, str, str]] = []
    try:
        paths = [path for path in task.context_dir.rglob("*") if path.is_file()]
    except OSError:
        return fields

    for path in sorted(paths):
        rel_path = path.relative_to(task.context_dir).as_posix()
        suffix = path.suffix.lower()
        try:
            if suffix in {".csv", ".tsv"}:
                delimiter = "\t" if suffix == ".tsv" else ","
                with path.open(newline="", encoding="utf-8-sig") as handle:
                    header = next(csv.reader(handle, delimiter=delimiter), [])
                for column in header:
                    if column:
                        fields.append((rel_path, "", str(column)))
            elif suffix in {".sqlite", ".sqlite3", ".db"}:
                with sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True) as conn:
                    table_rows = conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                    ).fetchall()
                    for (table_name,) in table_rows:
                        for column in conn.execute(f"PRAGMA table_info({table_name!r})").fetchall():
                            fields.append((rel_path, str(table_name), str(column[1])))
            elif suffix == ".json" and path.stat().st_size <= 2_000_000:
                payload = json.loads(path.read_text(encoding="utf-8"))
                table_name, columns = _json_record_fields(payload)
                for column in columns:
                    fields.append((rel_path, table_name or "", str(column)))
        except Exception:
            continue
    return fields


def _build_schema_match_report(task: PublicTask) -> str:
    fields = _scan_schema_fields(task)
    if not fields:
        return ""

    terms = _question_schema_terms(task.question)
    table_matches: list[str] = []
    field_matches: list[str] = []
    observed_lines: list[str] = []

    for rel_path, table_name, column in fields:
        location = f"{rel_path}"
        if table_name:
            location += f":{table_name}"
        normalized_table = _normalize_schema_name(table_name) if table_name else ""
        normalized_column = _normalize_schema_name(column)
        if normalized_table and normalized_table in terms:
            table_matches.append(f'- "{table_name}" -> {location}')
        if normalized_column in terms:
            field_matches.append(f'- "{column}" -> {location}.{column}')
        observed_lines.append(f"- {location}: {column}")

    max_observed = 80
    report_parts = [
        "Schema scan report:",
        "This is only observed schema metadata. Use exact/normalized field-name matches before semantic alternatives.",
    ]
    if field_matches:
        report_parts.append("Exact/normalized field matches from the question:")
        report_parts.extend(field_matches[:30])
    else:
        report_parts.append("Exact/normalized field matches from the question: none found")
    if table_matches:
        report_parts.append("Exact/normalized table/entity matches from the question:")
        report_parts.extend(table_matches[:30])
    report_parts.append("Observed fields:")
    report_parts.extend(observed_lines[:max_observed])
    if len(observed_lines) > max_observed:
        report_parts.append(f"- ... {len(observed_lines) - max_observed} more fields omitted")
    return "\n".join(report_parts) + "\n"


def build_system_prompt(tool_descriptions: str, system_prompt: str | None = None) -> str:
    base_prompt = system_prompt or REACT_SYSTEM_PROMPT
    return (
        f"{base_prompt}\n\n"
        "Available tools:\n"
        f"{tool_descriptions}\n\n"
        f"{RESPONSE_EXAMPLES}\n\n"
        "You must always return a single ```json fenced block containing one JSON object "
        "with keys `thought`, `action`, and `action_input`, and no extra text."
    )


def build_task_prompt(task: PublicTask) -> str:
    context_hint = _build_context_hint(task)
    schema_report = _build_schema_match_report(task)
    return (
        f"Question: {task.question}\n"
        f"{context_hint}"
        f"{schema_report}"
        "All tool file paths are relative to the task context directory. "
        "When you have the final table, call the `answer` tool."
    )


def build_observation_prompt(observation: dict[str, object]) -> str:
    rendered = json.dumps(observation, ensure_ascii=False, indent=2, default=str)
    return f"Observation:\n{rendered}"
