from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest


EVALUATE_PATH = Path(__file__).resolve().parents[1] / "evaluate.py"
EVALUATE_SPEC = importlib.util.spec_from_file_location("evaluate", EVALUATE_PATH)
assert EVALUATE_SPEC is not None
assert EVALUATE_SPEC.loader is not None
evaluate = importlib.util.module_from_spec(EVALUATE_SPEC)
sys.modules[EVALUATE_SPEC.name] = evaluate
EVALUATE_SPEC.loader.exec_module(evaluate)


def _write_csv(path: Path, columns: list[str], rows: list[list[object]]) -> Path:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    return path


def test_content_mode_ignores_column_names_but_strict_mode_does_not(tmp_path: Path) -> None:
    gold = _write_csv(tmp_path / "gold.csv", ["gold_name"], [[1], [2]])
    prediction = _write_csv(tmp_path / "prediction.csv", ["friendly_name"], [[1], [2]])

    content = evaluate._evaluate_csv(gold, prediction, mode="content")
    strict = evaluate._evaluate_csv(gold, prediction, mode="strict")

    assert content.matched
    assert content.score == pytest.approx(1.0)
    assert evaluate._compare_csv(gold, prediction)[0]
    assert strict.matched is False
    assert strict.note == "column mismatch"


def test_row_order_is_ignored_in_both_modes(tmp_path: Path) -> None:
    gold = _write_csv(tmp_path / "gold.csv", ["id", "label"], [[1, "A"], [2, "B"]])
    prediction = _write_csv(
        tmp_path / "prediction.csv", ["id", "label"], [[2, "B"], [1, "A"]]
    )

    assert evaluate._evaluate_csv(gold, prediction, mode="content").matched
    assert evaluate._evaluate_csv(gold, prediction, mode="strict").matched


def test_content_mode_allows_small_numeric_error(tmp_path: Path) -> None:
    gold = _write_csv(tmp_path / "gold.csv", ["metric"], [["1.000000000"]])
    prediction = _write_csv(tmp_path / "prediction.csv", ["metric"], [["1.000000001"]])

    content = evaluate._evaluate_csv(gold, prediction, mode="content")
    strict = evaluate._evaluate_csv(gold, prediction, mode="strict")

    assert content.matched
    assert strict.matched is False


def test_missing_gold_column_reduces_recall(tmp_path: Path) -> None:
    gold = _write_csv(tmp_path / "gold.csv", ["id", "label"], [[1, "A"], [2, "B"]])
    prediction = _write_csv(tmp_path / "prediction.csv", ["renamed_id"], [[1], [2]])

    result = evaluate._evaluate_csv(gold, prediction, mode="content")

    assert result.matched is False
    assert result.gold_column_recall == pytest.approx(0.5)
    assert result.row_f1 == pytest.approx(1.0)
    assert result.score == pytest.approx(0.5)


def test_extra_prediction_column_gets_light_penalty(tmp_path: Path) -> None:
    gold = _write_csv(tmp_path / "gold.csv", ["id"], [[1], [2]])
    prediction = _write_csv(
        tmp_path / "prediction.csv", ["renamed_id", "extra"], [[1, "x"], [2, "y"]]
    )

    result = evaluate._evaluate_csv(gold, prediction, mode="content")

    assert result.matched is False
    assert result.gold_column_recall == pytest.approx(1.0)
    assert result.row_f1 == pytest.approx(1.0)
    assert result.extra_prediction_columns == 1
    assert result.score == pytest.approx(0.95)


def test_wrong_content_and_wrong_row_associations_do_not_match(tmp_path: Path) -> None:
    gold = _write_csv(tmp_path / "gold.csv", ["id", "label"], [[1, "A"], [2, "B"]])
    wrong_content = _write_csv(
        tmp_path / "wrong-content.csv", ["id", "label"], [[1, "A"], [999, "B"]]
    )
    crossed_rows = _write_csv(
        tmp_path / "crossed-rows.csv", ["id", "label"], [[1, "B"], [2, "A"]]
    )

    content_result = evaluate._evaluate_csv(gold, wrong_content, mode="content")
    crossed_result = evaluate._evaluate_csv(gold, crossed_rows, mode="content")

    assert content_result.matched is False
    assert content_result.score < 1.0
    assert crossed_result.gold_column_recall == pytest.approx(1.0)
    assert crossed_result.row_f1 == pytest.approx(0.0)
    assert crossed_result.matched is False


def test_null_multiplicity_affects_content_match(tmp_path: Path) -> None:
    gold = _write_csv(tmp_path / "gold.csv", ["value"], [[None], [None], [1]])
    prediction = _write_csv(tmp_path / "prediction.csv", ["value"], [[None], [1], [1]])

    result = evaluate._evaluate_csv(gold, prediction, mode="content")

    assert result.matched is False
    assert result.gold_column_recall == pytest.approx(0.0)
    assert result.score == pytest.approx(0.0)


def test_strict_mode_preserves_legacy_comparison_rules(tmp_path: Path) -> None:
    gold = _write_csv(tmp_path / "gold.csv", [" Value "], [["1.0"], ["Hello   World"]])
    equivalent = _write_csv(
        tmp_path / "equivalent.csv", ["value"], [[" hello world "], ["1.00"]]
    )
    wrong_header = _write_csv(tmp_path / "wrong-header.csv", ["other"], [["1.0"], ["hello world"]])
    wrong_count = _write_csv(tmp_path / "wrong-count.csv", ["value"], [["1.0"]])
    wrong_content = _write_csv(
        tmp_path / "wrong-content.csv", ["value"], [["1.0"], ["different"]]
    )

    assert evaluate._evaluate_csv(gold, equivalent, mode="strict").matched

    header_result = evaluate._evaluate_csv(gold, wrong_header, mode="strict")
    assert header_result.matched is False
    assert header_result.note == "column mismatch"

    count_result = evaluate._evaluate_csv(gold, wrong_count, mode="strict")
    assert count_result.matched is False
    assert count_result.note == "row count: pred=1 gold=2"

    content_result = evaluate._evaluate_csv(gold, wrong_content, mode="strict")
    assert content_result.matched is False
    assert content_result.note == ""
