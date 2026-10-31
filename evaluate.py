from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import Literal, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = PROJECT_ROOT / "artifacts" / "runs"
DEFAULT_GOLD_DIR = PROJECT_ROOT / "public" / "output"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "public" / "input"

CONTENT_COLUMN_MATCH_THRESHOLD = 0.95
NUMERIC_ABS_TOLERANCE = Decimal("1e-9")
NUMERIC_REL_TOLERANCE = Decimal("1e-9")
EXTRA_COLUMN_PENALTY_WEIGHT = 0.1
MISSING_TEXT_VALUES = frozenset({"", "null", "none", "nan"})

EvaluationMode = Literal["content", "strict"]
NormalizedCell = tuple[str, Decimal | str | None]


@dataclass(frozen=True)
class ColumnMatch:
    gold_index: int
    prediction_index: int
    similarity: float


@dataclass(frozen=True)
class CsvEvaluation:
    matched: bool
    score: float
    note: str
    gold_column_recall: float
    row_f1: float
    extra_prediction_columns: int
    column_matches: tuple[ColumnMatch, ...] = ()


def _task_number(task_id: str) -> int:
    match = re.search(r"(\d+)$", task_id)
    return int(match.group(1)) if match else 0


def _latest_run_dir(runs_dir: Path = DEFAULT_RUNS_DIR) -> Path:
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Missing runs directory: {runs_dir}")
    candidates = [path for path in runs_dir.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under: {runs_dir}")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _difficulty(task_id: str) -> str:
    task_json_path = DEFAULT_INPUT_DIR / task_id / "task.json"
    if not task_json_path.exists():
        return "unknown"
    payload = json.loads(task_json_path.read_text())
    return str(payload.get("difficulty", "unknown"))


def _normalize_column(value: object) -> str:
    text = str(value).strip().lower()
    return re.sub(r"\s+", " ", text)


def _decimal_value(text: str) -> Decimal | None:
    cleaned = text.strip().replace(",", "")
    if not cleaned:
        return None
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1]
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def _normalize_cell(value: object) -> tuple[str, str]:
    """Legacy cell normalization retained for strict-mode compatibility."""
    text = str(value).strip()
    number = _decimal_value(text)
    if number is not None:
        return ("number", str(number.normalize()))
    text = re.sub(r"\s+", " ", text).strip().lower()
    return ("text", text)


def _normalized_rows(rows: list[list[str]]) -> Counter[tuple[tuple[str, str], ...]]:
    """Legacy row normalization retained for strict-mode compatibility."""
    return Counter(tuple(_normalize_cell(cell) for cell in row) for row in rows)


def _normalize_content_cell(value: object) -> NormalizedCell:
    text = unicodedata.normalize("NFKC", str(value)).strip()
    folded = re.sub(r"\s+", " ", text).casefold()
    if folded in MISSING_TEXT_VALUES:
        return ("missing", None)
    number = _decimal_value(text)
    if number is not None:
        return ("number", number)
    return ("text", folded)


def _numbers_equal(left: Decimal, right: Decimal) -> bool:
    tolerance = NUMERIC_ABS_TOLERANCE + NUMERIC_REL_TOLERANCE * max(
        abs(left), abs(right)
    )
    return abs(left - right) <= tolerance


def _content_cells_equal(left: NormalizedCell, right: NormalizedCell) -> bool:
    if left[0] != right[0]:
        return False
    if left[0] == "number":
        assert isinstance(left[1], Decimal)
        assert isinstance(right[1], Decimal)
        return _numbers_equal(left[1], right[1])
    return left[1] == right[1]


def _multiset_f1(overlap: int, gold_count: int, prediction_count: int) -> float:
    if gold_count == 0 and prediction_count == 0:
        return 1.0
    if overlap == 0 or gold_count == 0 or prediction_count == 0:
        return 0.0
    precision = overlap / prediction_count
    recall = overlap / gold_count
    return 2 * precision * recall / (precision + recall)


def _numeric_overlap(gold: list[Decimal], prediction: list[Decimal]) -> int:
    gold = sorted(gold)
    prediction = sorted(prediction)
    gold_index = 0
    prediction_index = 0
    overlap = 0
    while gold_index < len(gold) and prediction_index < len(prediction):
        gold_value = gold[gold_index]
        prediction_value = prediction[prediction_index]
        if _numbers_equal(gold_value, prediction_value):
            overlap += 1
            gold_index += 1
            prediction_index += 1
        elif gold_value < prediction_value:
            gold_index += 1
        else:
            prediction_index += 1
    return overlap


def _column_similarity(gold: Sequence[object], prediction: Sequence[object]) -> float:
    gold_cells = [_normalize_content_cell(value) for value in gold]
    prediction_cells = [_normalize_content_cell(value) for value in prediction]

    gold_missing = sum(cell[0] == "missing" for cell in gold_cells)
    prediction_missing = sum(cell[0] == "missing" for cell in prediction_cells)
    overlap = min(gold_missing, prediction_missing)

    gold_text = Counter(str(cell[1]) for cell in gold_cells if cell[0] == "text")
    prediction_text = Counter(str(cell[1]) for cell in prediction_cells if cell[0] == "text")
    overlap += sum((gold_text & prediction_text).values())

    gold_numbers = [cell[1] for cell in gold_cells if cell[0] == "number"]
    prediction_numbers = [cell[1] for cell in prediction_cells if cell[0] == "number"]
    assert all(isinstance(value, Decimal) for value in gold_numbers)
    assert all(isinstance(value, Decimal) for value in prediction_numbers)
    overlap += _numeric_overlap(gold_numbers, prediction_numbers)  # type: ignore[arg-type]

    return _multiset_f1(overlap, len(gold_cells), len(prediction_cells))


def _columns(rows: list[list[str]], column_count: int) -> list[list[str]]:
    return [
        [row[column_index] if column_index < len(row) else "" for row in rows]
        for column_index in range(column_count)
    ]


def _best_column_matches(
    similarities: list[list[float]],
    threshold: float = CONTENT_COLUMN_MATCH_THRESHOLD,
) -> tuple[ColumnMatch, ...]:
    gold_count = len(similarities)
    prediction_count = len(similarities[0]) if gold_count else 0

    @lru_cache(maxsize=None)
    def search(gold_index: int, used_prediction_mask: int) -> tuple[int, float, tuple[int, ...]]:
        if gold_index == gold_count:
            return (0, 0.0, ())

        skipped_count, skipped_score, skipped_assignment = search(
            gold_index + 1, used_prediction_mask
        )
        best = (skipped_count, skipped_score, (-1, *skipped_assignment))

        for prediction_index in range(prediction_count):
            if used_prediction_mask & (1 << prediction_index):
                continue
            similarity = similarities[gold_index][prediction_index]
            if similarity < threshold:
                continue
            matched_count, total_score, assignment = search(
                gold_index + 1, used_prediction_mask | (1 << prediction_index)
            )
            candidate = (
                matched_count + 1,
                total_score + similarity,
                (prediction_index, *assignment),
            )
            if candidate[:2] > best[:2]:
                best = candidate
        return best

    _, _, assignment = search(0, 0)
    return tuple(
        ColumnMatch(
            gold_index=gold_index,
            prediction_index=prediction_index,
            similarity=similarities[gold_index][prediction_index],
        )
        for gold_index, prediction_index in enumerate(assignment)
        if prediction_index >= 0
    )


def _rows_equal(left: tuple[NormalizedCell, ...], right: tuple[NormalizedCell, ...]) -> bool:
    return len(left) == len(right) and all(
        _content_cells_equal(left_cell, right_cell)
        for left_cell, right_cell in zip(left, right, strict=True)
    )


def _row_f1(
    gold_rows: list[list[str]],
    prediction_rows: list[list[str]],
    matches: tuple[ColumnMatch, ...],
) -> float:
    if not matches:
        return 1.0 if not gold_rows and not prediction_rows else 0.0

    ordered_matches = sorted(matches, key=lambda match: match.gold_index)
    normalized_gold = [
        tuple(
            _normalize_content_cell(
                row[match.gold_index] if match.gold_index < len(row) else ""
            )
            for match in ordered_matches
        )
        for row in gold_rows
    ]
    normalized_prediction = [
        tuple(
            _normalize_content_cell(
                row[match.prediction_index] if match.prediction_index < len(row) else ""
            )
            for match in ordered_matches
        )
        for row in prediction_rows
    ]

    prediction_match: dict[int, int] = {}

    def augment(gold_index: int, seen: set[int]) -> bool:
        for prediction_index, prediction_row in enumerate(normalized_prediction):
            if prediction_index in seen:
                continue
            if not _rows_equal(normalized_gold[gold_index], prediction_row):
                continue
            seen.add(prediction_index)
            previous_gold = prediction_match.get(prediction_index)
            if previous_gold is None or augment(previous_gold, seen):
                prediction_match[prediction_index] = gold_index
                return True
        return False

    overlap = sum(augment(gold_index, set()) for gold_index in range(len(normalized_gold)))
    return _multiset_f1(overlap, len(normalized_gold), len(normalized_prediction))


def _strict_compare(
    gold_columns: list[str],
    gold_rows: list[list[str]],
    prediction_columns: list[str],
    prediction_rows: list[list[str]],
) -> tuple[bool, str]:
    if [_normalize_column(column) for column in prediction_columns] != [
        _normalize_column(column) for column in gold_columns
    ]:
        return False, "column mismatch"
    if len(prediction_rows) != len(gold_rows):
        return False, f"row count: pred={len(prediction_rows)} gold={len(gold_rows)}"
    if _normalized_rows(prediction_rows) != _normalized_rows(gold_rows):
        return False, ""
    return True, ""


def _content_evaluation(
    gold_columns: list[str],
    gold_rows: list[list[str]],
    prediction_columns: list[str],
    prediction_rows: list[list[str]],
) -> CsvEvaluation:
    gold_values = _columns(gold_rows, len(gold_columns))
    prediction_values = _columns(prediction_rows, len(prediction_columns))
    similarities = [
        [
            _column_similarity(gold_column, prediction_column)
            for prediction_column in prediction_values
        ]
        for gold_column in gold_values
    ]
    column_matches = _best_column_matches(similarities)

    gold_column_recall = (
        len(column_matches) / len(gold_columns)
        if gold_columns
        else (1.0 if not prediction_columns else 0.0)
    )
    row_f1 = _row_f1(gold_rows, prediction_rows, column_matches)
    extra_prediction_columns = max(0, len(prediction_columns) - len(column_matches))
    extra_ratio = (
        extra_prediction_columns / len(prediction_columns) if prediction_columns else 0.0
    )
    extra_factor = 1.0 - EXTRA_COLUMN_PENALTY_WEIGHT * extra_ratio
    score = gold_column_recall * row_f1 * extra_factor
    matched = math.isclose(score, 1.0, rel_tol=0.0, abs_tol=1e-12)

    match_details = ", ".join(
        f"gold[{match.gold_index}]<-pred[{match.prediction_index}]"
        f"({match.similarity:.3f})"
        for match in column_matches
    )
    note = (
        f"gold_recall={gold_column_recall:.3f}; row_f1={row_f1:.3f}; "
        f"extra_columns={extra_prediction_columns}"
    )
    if match_details:
        note += f"; matches={match_details}"

    return CsvEvaluation(
        matched=matched,
        score=score,
        note=note,
        gold_column_recall=gold_column_recall,
        row_f1=row_f1,
        extra_prediction_columns=extra_prediction_columns,
        column_matches=column_matches,
    )


def _evaluate_csv(
    gold_path: Path,
    prediction_path: Path | None,
    *,
    mode: EvaluationMode = "content",
) -> CsvEvaluation:
    if prediction_path is None or not prediction_path.exists():
        return CsvEvaluation(
            matched=False,
            score=0.0,
            note="no prediction (agent failed)",
            gold_column_recall=0.0,
            row_f1=0.0,
            extra_prediction_columns=0,
        )

    gold_columns, gold_rows = _read_csv(gold_path)
    prediction_columns, prediction_rows = _read_csv(prediction_path)

    if mode == "content":
        return _content_evaluation(
            gold_columns, gold_rows, prediction_columns, prediction_rows
        )
    if mode == "strict":
        matched, note = _strict_compare(
            gold_columns, gold_rows, prediction_columns, prediction_rows
        )
        return CsvEvaluation(
            matched=matched,
            score=float(matched),
            note=note,
            gold_column_recall=float(matched),
            row_f1=float(matched),
            extra_prediction_columns=max(0, len(prediction_columns) - len(gold_columns)),
        )
    raise ValueError(f"Unknown evaluation mode: {mode}")


def _compare_csv(
    gold_path: Path,
    prediction_path: Path | None,
    *,
    mode: EvaluationMode = "content",
) -> tuple[bool, str]:
    result = _evaluate_csv(gold_path, prediction_path, mode=mode)
    return result.matched, result.note


def _resolve_run_dir(run_dir_arg: str | None) -> Path:
    if run_dir_arg:
        run_dir = Path(run_dir_arg)
        if not run_dir.is_absolute():
            run_dir = (PROJECT_ROOT / run_dir).resolve()
        return run_dir
    run_dir = _latest_run_dir()
    print(f"Auto-selected latest run: {run_dir.name}\n")
    return run_dir


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate prediction CSV files against Gold.")
    parser.add_argument("run_dir", nargs="?", help="Run directory; defaults to the latest run.")
    parser.add_argument(
        "--mode",
        choices=("content", "strict"),
        default="content",
        help="Evaluation mode (default: content).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dir = _resolve_run_dir(args.run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not DEFAULT_GOLD_DIR.is_dir():
        raise FileNotFoundError(f"Gold directory not found: {DEFAULT_GOLD_DIR}")

    task_ids = sorted((path.name for path in DEFAULT_GOLD_DIR.glob("task_*")), key=_task_number)

    rows: list[tuple[str, str, CsvEvaluation]] = []
    by_difficulty: dict[str, list[CsvEvaluation]] = defaultdict(list)
    no_prediction_count = 0

    for task_id in task_ids:
        difficulty = _difficulty(task_id)
        gold_path = DEFAULT_GOLD_DIR / task_id / "gold.csv"
        prediction_path = run_dir / task_id / "prediction.csv"
        result = _evaluate_csv(gold_path, prediction_path, mode=args.mode)
        if result.note.startswith("no prediction"):
            no_prediction_count += 1
        by_difficulty[difficulty].append(result)
        rows.append((task_id, difficulty, result))

    print(f"Evaluation mode: {args.mode}\n")
    print(
        f"{'Task':<12} {'Difficulty':<10} {'Match':<8} {'Score':<8} "
        f"{'Recall':<8} {'Row F1':<8} {'Extra':<6} Note"
    )
    print("-" * 110)
    for task_id, difficulty, result in rows:
        mark = "✓" if result.matched else "✗"
        print(
            f"{task_id:<12} {difficulty:<10} {mark:<8} {result.score:<8.3f} "
            f"{result.gold_column_recall:<8.3f} {result.row_f1:<8.3f} "
            f"{result.extra_prediction_columns:<6} {result.note}"
        )

    total_count = len(rows)
    evaluated_count = total_count - no_prediction_count
    correct_count = sum(result.matched for _, _, result in rows)
    total_score = sum(result.score for _, _, result in rows)
    print("\n" + "=" * 70)
    print(f"Total tasks:      {total_count}")
    print(f"Evaluated:        {evaluated_count}")
    print(f"Failed (no pred): {no_prediction_count}")
    if evaluated_count:
        print(
            f"Correct:          {correct_count} / {evaluated_count}  "
            f"({correct_count / evaluated_count:.1%} of evaluated)"
        )
    else:
        print("Correct:          0 / 0")
    if total_count:
        print(f"Overall accuracy: {correct_count / total_count:.1%}  (including failed tasks)")
        print(f"Average score:    {total_score / total_count:.3f}")

    print("\nBy difficulty:")
    for difficulty in ["easy", "medium", "hard", "extreme", "unknown"]:
        results = by_difficulty.get(difficulty)
        if not results:
            continue
        subtotal = len(results)
        subcorrect = sum(result.matched for result in results)
        average_score = sum(result.score for result in results) / subtotal
        print(
            f"  {difficulty:<8}: {subcorrect}/{subtotal}  "
            f"({subcorrect / subtotal:.0%}), score={average_score:.3f}"
        )

    return 0 if correct_count == total_count else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as exc:
        print(f"Evaluation unavailable: {exc}")
        raise SystemExit(2) from None
