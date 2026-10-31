from __future__ import annotations

import csv
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from data_agent_baseline.benchmark.dataset import DABenchPublicDataset
from data_agent_baseline.benchmark.schema import PublicTask
from data_agent_baseline.semantic.catalog import SemanticCatalog, load_semantic_catalog
from data_agent_baseline.semantic.schema import SemanticModel


_DESCRIPTION_STOPWORDS = {
    "and",
    "as",
    "calendar",
    "commonly",
    "derived",
    "from",
    "governed",
    "identifier",
    "name",
    "or",
    "such",
    "the",
    "unique",
}


def _words(value: str) -> tuple[str, ...]:
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    words: list[str] = []
    for raw in re.findall(r"[A-Za-z0-9]+", expanded.casefold()):
        if raw.endswith("oes") and len(raw) > 4:
            raw = raw[:-2]
        elif raw.endswith("ies") and len(raw) > 4:
            raw = raw[:-3] + "y"
        elif raw.endswith("s") and len(raw) > 3 and not raw.endswith("ss"):
            raw = raw[:-1]
        words.append(raw)
    return tuple(words)


def _word_is_present(word: str, question_words: set[str]) -> bool:
    if word in question_words:
        return True
    return len(word) >= 4 and any(
        question_word.endswith(word) or word.endswith(question_word)
        for question_word in question_words
        if len(question_word) >= 4
    )


def _phrase_is_present(
    words: tuple[str, ...],
    question_words: set[str],
    *,
    ignore_id: bool = True,
) -> bool:
    relevant = tuple(word for word in words if not (ignore_id and word == "id"))
    return bool(relevant) and all(_word_is_present(word, question_words) for word in relevant)


def _quote_sqlite_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _csv_signature(path: Path) -> tuple[tuple[str, ...], str | None]:
    try:
        with path.open(newline="") as handle:
            header = next(csv.reader(handle), [])
        return tuple(str(item) for item in header), None
    except Exception as exc:  # noqa: BLE001
        return (), f"{type(exc).__name__}: {exc}"


def _json_signature(path: Path) -> tuple[tuple[str, ...], tuple[str, ...], str | None]:
    try:
        payload = json.loads(path.read_text())
        relations: tuple[str, ...] = ()
        records: object = payload
        if isinstance(payload, dict):
            if payload.get("table") is not None:
                relations = (str(payload["table"]),)
            records = payload.get("records", payload)
        fields: set[str] = set()
        if isinstance(records, list):
            for item in records:
                if isinstance(item, dict):
                    fields.update(str(key) for key in item)
        elif isinstance(records, dict):
            fields.update(str(key) for key in records)
        return relations, tuple(sorted(fields)), None
    except Exception as exc:  # noqa: BLE001
        return (), (), f"{type(exc).__name__}: {exc}"


def _sqlite_signature(
    path: Path,
) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...], str | None]:
    try:
        uri = f"file:{path.resolve().as_posix()}?mode=ro"
        relations: list[dict[str, Any]] = []
        all_fields: set[str] = set()
        with sqlite3.connect(uri, uri=True) as conn:
            table_rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            ).fetchall()
            for (table_name,) in table_rows:
                columns = tuple(
                    str(row[1])
                    for row in conn.execute(
                        f"PRAGMA table_info({_quote_sqlite_identifier(str(table_name))})"
                    ).fetchall()
                )
                all_fields.update(columns)
                relations.append({"name": str(table_name), "fields": list(columns)})
        return tuple(relations), tuple(sorted(all_fields)), None
    except Exception as exc:  # noqa: BLE001
        return (), (), f"{type(exc).__name__}: {exc}"


def build_source_signature(task: PublicTask) -> tuple[dict[str, Any], ...]:
    assets: list[dict[str, Any]] = []
    for path in sorted(item for item in task.context_dir.rglob("*") if item.is_file()):
        relative_path = path.relative_to(task.context_dir).as_posix()
        suffix = path.suffix.casefold()
        asset: dict[str, Any] = {
            "path": relative_path,
            "kind": "file",
            "relations": [],
            "fields": [],
        }
        if suffix in {".csv", ".tsv"}:
            fields, error = _csv_signature(path)
            asset.update(kind="csv", relations=[path.stem], fields=list(fields))
            if error:
                asset["inspection_error"] = error
        elif suffix == ".json":
            relations, fields, error = _json_signature(path)
            asset.update(
                kind="json", relations=list(relations or (path.stem,)), fields=list(fields)
            )
            if error:
                asset["inspection_error"] = error
        elif suffix in {".db", ".sqlite", ".sqlite3"}:
            relations, fields, error = _sqlite_signature(path)
            asset.update(kind="sqlite", relations=list(relations), fields=list(fields))
            if error:
                asset["inspection_error"] = error
        assets.append(asset)
    return tuple(assets)


def _model_expression_fields(model: SemanticModel) -> set[str]:
    identifiers: set[str] = set()
    expressions = [item.expression for item in model.dimensions.values()]
    expressions.extend(item.expression for item in model.metrics.values())
    expressions.extend(item.condition for item in model.joins)
    for expression in expressions:
        identifiers.update(
            item.casefold() for item in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression)
        )
    return identifiers


def _mentioned_source_fields(
    question: str,
    signature: tuple[dict[str, Any], ...],
) -> set[str]:
    question_words = set(_words(question))
    mentioned: set[str] = set()
    for asset in signature:
        for field in asset.get("fields", []):
            field_name = str(field)
            if _phrase_is_present(_words(field_name), question_words):
                mentioned.add(field_name)
    normalized = {field.casefold() for field in mentioned}
    return {
        field
        for field in mentioned
        if not (field.casefold().endswith("_id") and field.casefold()[:-3] in normalized)
    }


def _capability_aliases(name: str, output_label: str) -> tuple[tuple[str, ...], ...]:
    full_name = _words(name)
    suffix = _words(name.rsplit(".", 1)[-1])
    output = _words(output_label)
    aliases = {full_name, output}
    if suffix and suffix not in {("id",), ("name",)}:
        aliases.add(suffix)
    return tuple(alias for alias in aliases if alias)


def _description_indicates_filter(
    description: str,
    capability_name: str,
    output_label: str,
    question_words: set[str],
) -> bool:
    semantic_words = set(_words(capability_name)) | set(_words(output_label))
    description_words = {
        word
        for word in _words(description)
        if word not in _DESCRIPTION_STOPWORDS and word not in semantic_words
    }
    matched_words = {word for word in description_words if _word_is_present(word, question_words)}
    uppercase_values = {
        value.casefold() for value in re.findall(r"\b[A-Z][A-Z0-9_-]{2,}\b", description)
    }
    return bool(uppercase_values & question_words) or len(matched_words) >= 2


def infer_model_capabilities(task: PublicTask, model: SemanticModel) -> dict[str, list[str]]:
    question_words = set(_words(task.question))
    dimensions: set[str] = set()
    filters: set[str] = set()
    metrics: set[str] = set()
    for name, dimension in model.dimensions.items():
        aliases = _capability_aliases(name, dimension.output_label)
        if any(_phrase_is_present(alias, question_words, ignore_id=False) for alias in aliases):
            dimensions.add(name)
        if _description_indicates_filter(
            dimension.description,
            name,
            dimension.output_label,
            question_words,
        ):
            filters.add(name)
    for name, metric in model.metrics.items():
        aliases = _capability_aliases(name, metric.output_label)
        if any(_phrase_is_present(alias, question_words, ignore_id=False) for alias in aliases):
            metrics.add(name)
    return {
        "dimensions": sorted(dimensions),
        "metrics": sorted(metrics),
        "filter_fields": sorted(filters),
        "order_fields": [],
    }


def _unmapped_mentioned_fields(
    task: PublicTask,
    model: SemanticModel,
    signature: tuple[dict[str, Any], ...],
) -> list[str]:
    exposed_identifiers = _model_expression_fields(model)
    mentioned = _mentioned_source_fields(task.question, signature)
    return sorted(field for field in mentioned if field.casefold() not in exposed_identifiers)


def scan_semantic_coverage(
    tasks_root: Path,
    models_dir: Path,
) -> dict[str, Any]:
    dataset = DABenchPublicDataset(tasks_root)
    catalog: SemanticCatalog = load_semantic_catalog(models_dir)
    task_results: list[dict[str, Any]] = []
    source_candidate_count = 0
    matched_model_count = 0
    capability_rejection_count = 0
    for task in dataset.iter_tasks():
        signature = build_source_signature(task)
        decisions: list[dict[str, Any]] = []
        matched_models: list[str] = []
        for model in catalog.models:
            path_decision = next(
                item
                for item in catalog.evaluate_task(task)
                if item.model.model_id == model.model_id
            )
            capabilities = {
                "dimensions": [],
                "metrics": [],
                "filter_fields": [],
                "order_fields": [],
            }
            decision = path_decision
            unmapped_fields: list[str] = []
            if path_decision.source_compatible:
                source_candidate_count += 1
                capabilities = infer_model_capabilities(task, model)
                decision = next(
                    item
                    for item in catalog.evaluate_task(
                        task,
                        required_dimensions=capabilities["dimensions"],
                        required_metrics=capabilities["metrics"],
                        required_filter_fields=capabilities["filter_fields"],
                        required_order_fields=capabilities["order_fields"],
                    )
                    if item.model.model_id == model.model_id
                )
            reasons = list(decision.reasons)
            matched = decision.matched
            if decision.source_compatible:
                unmapped_fields = _unmapped_mentioned_fields(task, model, signature)
                if unmapped_fields:
                    matched = False
                    reasons.append(
                        "question references source fields not exposed by model: "
                        + ", ".join(unmapped_fields)
                    )
                if matched:
                    matched_model_count += 1
                    matched_models.append(model.model_id)
                else:
                    capability_rejection_count += 1
            decision_payload = decision.to_dict()
            decision_payload.update(
                matched=matched,
                reasons=reasons,
                inferred_capabilities=capabilities,
                unmapped_source_fields=unmapped_fields,
            )
            decisions.append(decision_payload)
        task_results.append(
            {
                "task_id": task.task_id,
                "difficulty": task.difficulty,
                "question": task.question,
                "source_signature": list(signature),
                "matched_models": matched_models,
                "decisions": decisions,
            }
        )
    matched_task_count = sum(bool(item["matched_models"]) for item in task_results)
    return {
        "tasks_root": str(tasks_root.resolve()),
        "models_dir": str(models_dir.resolve()),
        "summary": {
            "task_count": len(task_results),
            "model_count": len(catalog.models),
            "source_candidate_count": source_candidate_count,
            "matched_task_count": matched_task_count,
            "unmatched_task_count": len(task_results) - matched_task_count,
            "matched_model_count": matched_model_count,
            "capability_rejection_count": capability_rejection_count,
        },
        "tasks": task_results,
    }


def _markdown_escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _render_source_signature(assets: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for asset in assets:
        relations: list[str] = []
        for relation in asset.get("relations", []):
            if isinstance(relation, dict):
                relations.append(str(relation.get("name", "")))
            else:
                relations.append(str(relation))
        relation_text = ",".join(item for item in relations if item)
        detail = f"{asset['kind']}"
        if relation_text:
            detail += f":{relation_text}"
        rendered.append(f"`{asset['path']}` ({detail})")
    return "<br>".join(rendered)


def _render_capabilities(decision: dict[str, Any]) -> str:
    inferred = decision["inferred_capabilities"]
    parts = [
        f"dimensions={','.join(inferred['dimensions']) or '-'}",
        f"metrics={','.join(inferred['metrics']) or '-'}",
        f"filters={','.join(inferred['filter_fields']) or '-'}",
    ]
    unmapped = decision.get("unmapped_source_fields", [])
    if unmapped:
        parts.append(f"unmapped={','.join(unmapped)}")
    return "; ".join(parts)


def render_coverage_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Semantic Coverage Report",
        "",
        f"- Tasks root: `{report['tasks_root']}`",
        f"- Models directory: `{report['models_dir']}`",
        f"- Tasks: {summary['task_count']}",
        f"- Semantic models: {summary['model_count']}",
        f"- Source-compatible candidates: {summary['source_candidate_count']}",
        f"- Matched tasks: {summary['matched_task_count']}",
        f"- Unmatched tasks: {summary['unmatched_task_count']}",
        f"- Capability rejections after source match: {summary['capability_rejection_count']}",
        "",
        "Scanner capability inference is a conservative static preflight based on the question, "
        "declared semantic names/descriptions, and explicit source field names. Runtime matching "
        "uses the submitted ASQ as the authoritative capability requirement.",
        "",
        "| Task | Difficulty | Question | Source signature | Model decisions |",
        "| --- | --- | --- | --- | --- |",
    ]
    for task in report["tasks"]:
        model_decisions: list[str] = []
        for decision in task["decisions"]:
            status = "MATCH" if decision["matched"] else "REJECT"
            reasons = "; ".join(decision["reasons"]) or "all inferred capabilities covered"
            capabilities = _render_capabilities(decision)
            model_decisions.append(
                f"**{status} {decision['model_id']}** — {capabilities}; {reasons}"
            )
        lines.append(
            "| "
            + " | ".join(
                (
                    _markdown_escape(task["task_id"]),
                    _markdown_escape(task["difficulty"]),
                    _markdown_escape(task["question"]),
                    _markdown_escape(_render_source_signature(task["source_signature"])),
                    _markdown_escape("<br>".join(model_decisions)),
                )
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def write_coverage_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_coverage_report(report))
