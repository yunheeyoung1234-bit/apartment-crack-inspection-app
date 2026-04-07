from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


OUTPUT_CSV_PATH = Path("inspection_records.csv")
CSV_HEADERS = [
    "filename",
    "timestamp",
    "detected_issue_count",
    "predicted_defect_type",
    "top_predicted_class",
    "confidence_score",
    "p25_width_px",
    "p25_width_mm",
    "auto_severity",
    "스케일입력방식",
    "적용 mm_per_px",
    "inspector_comment",
]


def save_inspection_record(
    filename: str,
    detected_issue_count: int,
    predicted_defect_type: str,
    top_predicted_class: str,
    confidence_score: float,
    p25_width_px: float | None,
    p25_width_mm: float | None,
    auto_severity: str,
    scale_input_method: str,
    applied_mm_per_px: float,
    inspector_comment: str,
    timestamp: str | None = None,
    csv_path: Path = OUTPUT_CSV_PATH,
) -> Path:
    csv_exists = csv_path.exists()
    record_timestamp = timestamp or datetime.now().isoformat(timespec="seconds")

    with csv_path.open("a", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADERS)
        if not csv_exists:
            writer.writeheader()
        writer.writerow(
            {
                "filename": filename,
                "timestamp": record_timestamp,
                "detected_issue_count": detected_issue_count,
                "predicted_defect_type": predicted_defect_type,
                "top_predicted_class": top_predicted_class,
                "confidence_score": f"{confidence_score:.4f}",
                "p25_width_px": _format_optional_number(p25_width_px),
                "p25_width_mm": _format_optional_number(p25_width_mm),
                "auto_severity": auto_severity,
                "스케일입력방식": scale_input_method,
                "적용 mm_per_px": f"{applied_mm_per_px:.6f}",
                "inspector_comment": inspector_comment.strip(),
            }
        )

    return csv_path


def _format_optional_number(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"
