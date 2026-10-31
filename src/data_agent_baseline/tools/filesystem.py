from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from data_agent_baseline.benchmark.schema import PublicTask


def resolve_context_path(task: PublicTask, relative_path: str) -> Path:
    candidate = (task.context_dir / relative_path).resolve()
    context_root = task.context_dir.resolve()
    if context_root not in candidate.parents and candidate != context_root:
        raise ValueError(f"Path escapes context dir: {relative_path}")
    if not candidate.exists():
        raise FileNotFoundError(f"Missing context asset: {relative_path}")
    return candidate


def list_context_tree(task: PublicTask, *, max_depth: int = 4) -> dict[str, object]:
    entries: list[dict[str, object]] = []

    def walk(path: Path, depth: int) -> None:
        if depth > max_depth:
            return
        for child in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name)):
            rel_path = child.relative_to(task.context_dir).as_posix()
            entries.append(
                {
                    "path": rel_path,
                    "kind": "dir" if child.is_dir() else "file",
                    "size": child.stat().st_size if child.is_file() else None,
                }
            )
            if child.is_dir():
                walk(child, depth + 1)

    walk(task.context_dir, 1)
    return {
        "root": str(task.context_dir),
        "entries": entries,
    }


def read_csv_preview(
    task: PublicTask, relative_path: str, *, max_rows: int = 20
) -> dict[str, object]:
    path = resolve_context_path(task, relative_path)
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return {
                "path": relative_path,
                "columns": [],
                "rows": [],
                "row_count": 0,
            }

        preview_rows: list[list[str]] = []
        row_count = 0
        for row in reader:
            row_count += 1
            if len(preview_rows) < max_rows:
                preview_rows.append(row)

    if not header:
        return {
            "path": relative_path,
            "columns": [],
            "rows": preview_rows,
            "row_count": row_count,
        }

    return {
        "path": relative_path,
        "columns": header,
        "rows": preview_rows,
        "row_count": row_count,
    }


def read_json_preview(
    task: PublicTask, relative_path: str, *, max_chars: int = 4000
) -> dict[str, object]:
    path = resolve_context_path(task, relative_path)
    payload = json.loads(path.read_text())
    preview = json.dumps(payload, ensure_ascii=False, indent=2)
    return {
        "path": relative_path,
        "preview": preview[:max_chars],
        "truncated": len(preview) > max_chars,
    }


def read_doc_preview(
    task: PublicTask,
    relative_path: str,
    *,
    max_chars: int = 4000,
    start: int = 0,
) -> dict[str, object]:
    path = resolve_context_path(task, relative_path)
    text = path.read_text(errors="replace")
    safe_start = max(0, min(start, len(text)))
    preview = text[safe_start : safe_start + max_chars]
    return {
        "path": relative_path,
        "start": safe_start,
        "preview": preview,
        "truncated": safe_start + len(preview) < len(text),
        "total_chars": len(text),
        "large_file": len(text) > 50000,
        "hint": (
            "Large text file: use search_context for relevant keywords or execute_python "
            "to compute a small final aggregate. Avoid sequentially reading the whole file."
            if len(text) > 50000
            else None
        ),
    }


TEXT_SEARCH_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".txt",
    ".tsv",
}


def search_context_text(
    task: PublicTask,
    query: str,
    *,
    path: str | None = None,
    regex: bool = False,
    case_sensitive: bool = False,
    max_matches: int = 20,
    context_chars: int = 120,
) -> dict[str, object]:
    if not query:
        raise ValueError("query must not be empty.")

    root = resolve_context_path(task, path) if path else task.context_dir
    if root.is_file():
        files = [root]
    else:
        files = [
            item
            for item in sorted(root.rglob("*"))
            if item.is_file() and item.suffix.lower() in TEXT_SEARCH_SUFFIXES
        ]

    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(query if regex else re.escape(query), flags=flags)
    matches: list[dict[str, object]] = []
    max_matches = max(1, min(max_matches, 100))
    context_chars = max(20, min(context_chars, 500))

    for file_path in files:
        text = file_path.read_text(errors="replace")
        line_starts = [0]
        for match in re.finditer("\n", text):
            line_starts.append(match.end())
        for match in pattern.finditer(text):
            line_number = 1
            for index, line_start in enumerate(line_starts):
                if line_start > match.start():
                    break
                line_number = index + 1
            snippet_start = max(0, match.start() - context_chars)
            snippet_end = min(len(text), match.end() + context_chars)
            matches.append(
                {
                    "path": file_path.resolve().relative_to(task.context_dir.resolve()).as_posix(),
                    "line": line_number,
                    "char_start": match.start(),
                    "char_end": match.end(),
                    "snippet": text[snippet_start:snippet_end].replace("\n", "\\n"),
                }
            )
            if len(matches) >= max_matches:
                return {
                    "query": query,
                    "regex": regex,
                    "matches": matches,
                    "truncated": True,
                }

    return {
        "query": query,
        "regex": regex,
        "matches": matches,
        "truncated": False,
    }


def _paragraph_record_ids(paragraph: str) -> list[str]:
    id_value = r"(?:rec[A-Za-z0-9]{8,}|[A-Za-z]*\d[A-Za-z0-9_-]*)"
    id_label = (
        r"(?:ID|identifier|registration number|registry number|registry code|"
        r"registration ID|registry ID|reference code|reference ID|reference number|"
        r"tracking code|tracking number|file number|code|key)"
    )
    patterns = [
        rf"\b(?:Registry Ref|Registry ID|Registration ID|Tracking Code|Tracking Number|"
        rf"file number):?\s*({id_value})\b",
        rf"\((?:ID|Registry Ref|Registry ID|Registration ID|Tracking Code):\s*({id_value})\)",
        rf"\b(?:subject|individual|case|asset|unit|entry|program|stakeholder)\s+(?:identified|registered|cataloged|tracked|listed)\s+(?:as|under|with|by)?\s*(?:the\s+)?{id_label}?\s*({id_value})\b",
        rf"\b(?:record|profile|summary|file|case|assessment|panel|analysis|history)\s+(?:for|of)\s+({id_value})\b",
        rf"\b(?:operative|asset|unit|subject|individual|stakeholder|entry)\s+({id_value})\b",
        rf"\b(?:filed under|registered under|registered with|registered at|tracked with|tracked under|"
        rf"cataloged with|cataloged under|cataloged at|maintained under|identified by|referenced by|"
        rf"on file with|linked to)\s+(?:the\s+)?(?:unique\s+)?{id_label}?\s*({id_value})\b",
        rf"\b(?:whose entry is|entry is|which is|who is|the operative is)\s+referenced by\s+{id_label}\s*({id_value})\b",
        rf"\b(?:with|under|at)\s+(?:the\s+)?(?:unique\s+)?{id_label}\s*({id_value})\b",
        rf"\b{id_label}\s*(?:is|of|:)?\s*({id_value})\b",
    ]
    record_ids: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for match in re.findall(pattern, paragraph, flags=re.IGNORECASE):
            record_id = str(match)
            if not re.search(r"\d", record_id):
                continue
            if record_id not in seen:
                seen.add(record_id)
                record_ids.append(record_id)
    return record_ids


def _normalise_field_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _field_tokens(field: str) -> list[str]:
    return [token for token in re.split(r"[^A-Za-z0-9]+", field) if token]


def _alias_regex(alias: str) -> str:
    tokens = _field_tokens(alias)
    if not tokens:
        return re.escape(alias)
    joined = r"[\s_\-/]+".join(re.escape(token) for token in tokens)
    return rf"(?<![A-Za-z0-9]){joined}(?![A-Za-z0-9])"


def _document_acronym_aliases(text: str, aliases: set[str]) -> set[str]:
    expanded = set(aliases)
    for alias in list(aliases):
        compact = re.sub(r"[^A-Za-z0-9]", "", alias)
        if not (2 <= len(compact) <= 8):
            continue
        pattern = rf"\b([A-Za-z][A-Za-z\s\-/]{{2,80}}?)\s*\(\s*{re.escape(alias)}\s*\)"
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            phrase = re.sub(r"\s+", " ", match.group(1)).strip(" ,.;:")
            phrase = re.split(r"\s+(?:and|or)\s+", phrase)[-1].strip(" ,.;:")
            phrase = re.sub(r"^(?:and|or|the|a|an)\s+", "", phrase, flags=re.IGNORECASE)
            if phrase:
                expanded.add(phrase)
    return expanded


def _field_aliases(field: str, text: str) -> set[str]:
    raw = field.strip()
    normalised = _normalise_field_name(raw)
    aliases = {
        raw,
        normalised,
        normalised.replace("_", " "),
        normalised.replace("_", "-"),
        normalised.replace("_", "/"),
    }
    tokens = _field_tokens(raw)
    unit_suffixes = {"cm", "kg", "mg", "dl", "ul", "u", "l", "percent", "percentage"}
    content_tokens = [token for token in tokens if token.lower() not in unit_suffixes]
    if content_tokens and content_tokens != tokens:
        aliases.add(" ".join(content_tokens))
        aliases.add("_".join(content_tokens))
    if content_tokens and content_tokens[-1].lower() in {"id", "code", "key", "number"}:
        aliases.add(" ".join(content_tokens[:-1]))
        aliases.add("_".join(content_tokens[:-1]))
    compact = re.sub(r"[^A-Za-z0-9]", "", raw)
    if 2 <= len(compact) <= 8:
        aliases.add(compact)
        aliases.add(compact.upper())
    generic_aliases = {
        "birth_year": {"birth year", "year of birth", "birthdate", "birthday", "born"},
        "birthday_year": {"birth year", "year of birth", "birthdate", "birthday", "born"},
        "year_of_birth": {"birth year", "year of birth", "birthdate", "birthday", "born"},
        "sex": {"sex", "gender", "male", "female"},
        "gender": {"sex", "gender", "male", "female"},
        "height_cm": {"height"},
        "weight_kg": {"weight", "mass"},
    }
    aliases.update(generic_aliases.get(normalised, set()))
    return {alias for alias in _document_acronym_aliases(text, aliases) if alias}


def _sentences_with_terms(paragraph: str, terms: tuple[str, ...]) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    return [
        sentence
        for sentence in sentences
        if any(re.search(term, sentence, re.IGNORECASE) for term in terms)
    ]


def _sentences_with_alias(paragraph: str, aliases: set[str]) -> list[str]:
    patterns = tuple(_alias_regex(alias) for alias in aliases)
    return _sentences_with_terms(paragraph, patterns)


def _number_value(text: str) -> float | int | None:
    if text.lower() in {"nan", "none", "null"}:
        return None
    value = float(text)
    return int(value) if value.is_integer() else value


def _extract_numeric_field(paragraph: str, field: str, aliases: set[str]) -> float | int | None:
    normalised = _normalise_field_name(field)
    number_pattern = r"(?<![A-Za-z0-9])(?:NaN|None|null|-?\d+(?:\.\d+)?)(?![A-Za-z0-9])"
    final_cue = (
        r"\b(?:corrected|revised|adjusted|confirmed|verified|rectified|finalized|final|"
        r"amended|accurate|definitive|precisely measured|ground truth|manual verification)\b"
    )
    initial_cue = (
        r"\b(?:initial|initially|preliminary|originally|mistaken|mistakenly|misread|"
        r"incorrect|error|early|first-pass|first reported|first logged|thought to be)\b"
    )
    candidates: list[tuple[int, int, int, float | int]] = []
    for sentence_index, sentence in enumerate(_sentences_with_alias(paragraph, aliases)):
        alias_matches = [
            match
            for alias in aliases
            for match in re.finditer(_alias_regex(alias), sentence, flags=re.IGNORECASE)
        ]
        if not alias_matches:
            continue
        for alias_match in alias_matches:
            window_start = max(0, alias_match.start() - 80)
            window_end = min(len(sentence), alias_match.end() + 520)
            window = sentence[window_start:window_end]
            for number_match in re.finditer(number_pattern, window, flags=re.IGNORECASE):
                value = _number_value(number_match.group(0))
                if value is None:
                    continue
                absolute_pos = window_start + number_match.start()
                before = window[max(0, number_match.start() - 160) : number_match.start()].lower()
                score = 0
                if re.search(final_cue, before):
                    score += 8
                if re.search(initial_cue, before):
                    score -= 4
                if normalised.endswith(("_id", "_code", "_key")):
                    score += 1 if str(number_match.group(0)).isdigit() else -1
                distance = abs(absolute_pos - alias_match.end())
                candidates.append((score, sentence_index, -distance, value))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates[-1][3]


def _extract_birth_year(paragraph: str) -> int | None:
    years: list[tuple[int, int]] = []
    for sentence_index, sentence in enumerate(
        _sentences_with_terms(
            paragraph, (r"\bborn\b", r"\bbirthdate\b", r"\bbirthday\b", r"\byear of birth\b")
        )
    ):
        sentence_years = [int(year) for year in re.findall(r"\b(19\d{2}|20\d{2})\b", sentence)]
        if not sentence_years:
            continue
        if re.search(
            r"\b(?:corrected|rectified|amended|confirmed|verified|final)\b", sentence, re.IGNORECASE
        ):
            years.append((2, sentence_years[-1]))
        else:
            years.append((1, sentence_years[0]))
    if not years:
        return None
    years.sort(key=lambda item: item[0])
    return years[-1][1]


def _extract_sex(paragraph: str) -> str | None:
    if re.search(
        r"\b(?:female individual|she is female|her status as female|identified as female|registered as female)\b",
        paragraph,
        re.IGNORECASE,
    ):
        return "F"
    if re.search(r"\bfemale\b", paragraph, re.IGNORECASE):
        return "F"
    if re.search(
        r"\b(?:male individual|he is male|his status as male|identified as male|registered as male)\b",
        paragraph,
        re.IGNORECASE,
    ):
        return "M"
    if re.search(r"\bmale\b", paragraph, re.IGNORECASE):
        return "M"
    return None


def _extract_text_field(paragraph: str, aliases: set[str]) -> str | None:
    for sentence in _sentences_with_alias(paragraph, aliases):
        for alias in aliases:
            pattern = (
                rf"{_alias_regex(alias)}[^.;:]{{0,80}}?"
                r"(?:is|was|as|to|:)\s+['\"]?([A-Z][A-Za-z0-9' .&/-]{1,80}|None|-)"
            )
            match = re.search(pattern, sentence, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip(" '\".,;:")
    return None


def _extract_field_value(paragraph: str, field: str, aliases: set[str]) -> object | None:
    normalised = _normalise_field_name(field)
    if normalised in {"birth_year", "birthday_year", "birthyear", "year_of_birth", "yob"}:
        return _extract_birth_year(paragraph)
    if normalised in {"sex", "gender"}:
        return _extract_sex(paragraph)
    numeric = _extract_numeric_field(paragraph, field, aliases)
    if numeric is not None:
        return numeric
    return _extract_text_field(paragraph, aliases)


def _record_id_cell(record_id: str) -> int | str:
    return int(record_id) if record_id.isdigit() else record_id


def _record_sort_key(record_id: str) -> tuple[int, int | str]:
    return (0, int(record_id)) if record_id.isdigit() else (1, record_id)


def _discover_document_fields(text: str) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()

    def add(field: str) -> None:
        normalised = _normalise_field_name(field)
        if normalised and normalised not in seen:
            seen.add(normalised)
            fields.append(normalised)

    if re.search(r"\b(?:born|birthdate|birthday|year of birth)\b", text, re.IGNORECASE):
        add("birth_year")
    if re.search(r"\b(?:male|female|sex|gender)\b", text, re.IGNORECASE):
        add("sex")
    acronym_pattern = r"\b([A-Za-z][A-Za-z-]+(?:\s+[A-Za-z][A-Za-z-]+){0,5})\s*\(\s*([A-Za-z][A-Za-z0-9-]{1,8})\s*\)"
    for _, acronym in re.findall(acronym_pattern, text[:50000]):
        if len(fields) >= 20:
            break
        if (
            acronym.lower() not in {"nan", "none", "null"}
            and sum(char.isupper() for char in acronym) >= 2
        ):
            add(acronym)
    return fields[:20]


def extract_document_records(
    task: PublicTask,
    relative_path: str,
    *,
    fields: list[str] | None = None,
    max_records: int = 1000,
    include_partial: bool = False,
) -> dict[str, object]:
    path = resolve_context_path(task, relative_path)
    text = path.read_text(errors="replace")
    requested_fields = list(dict.fromkeys(fields or _discover_document_fields(text)))
    records_by_id: dict[str, dict[str, object]] = {}
    alias_by_field = {field: _field_aliases(field, text) for field in requested_fields}

    paragraphs = [
        paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()
    ]
    for paragraph in paragraphs:
        record_ids = _paragraph_record_ids(paragraph)
        if not record_ids:
            continue

        extracted_values: dict[str, object] = {}
        for field, aliases in alias_by_field.items():
            value = _extract_field_value(paragraph, field, aliases)
            if value is not None:
                extracted_values[field] = value

        if not extracted_values:
            continue
        for record_id in record_ids:
            record = records_by_id.setdefault(record_id, {"id": _record_id_cell(record_id)})
            record.update(extracted_values)

    columns = ["id"] + requested_fields
    extracted_rows: list[list[object]] = []
    partial_record_count = 0
    for _, record in sorted(records_by_id.items(), key=lambda item: _record_sort_key(item[0])):
        has_any_requested = any(column in record for column in requested_fields)
        if not has_any_requested:
            continue
        has_all_requested = all(record.get(column) is not None for column in requested_fields)
        if has_all_requested or include_partial:
            extracted_rows.append([record.get(column) for column in columns])
        if not has_all_requested:
            partial_record_count += 1

    max_records = max(1, min(max_records, 5000))
    rows = extracted_rows[:max_records]
    return {
        "path": relative_path,
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "total_records": len(extracted_rows),
        "partial_record_count": partial_record_count,
        "include_partial": include_partial,
        "note": (
            "Rows include only records with non-null values for every requested field. "
            "Use include_partial=true only when missing values are needed."
            if not include_partial and partial_record_count
            else None
        ),
        "truncated": len(extracted_rows) > max_records,
    }
