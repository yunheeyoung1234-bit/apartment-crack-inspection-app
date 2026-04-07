from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw

from src.width_estimation import (
    MM_PER_PX,
    estimate_crack_width,
    get_preliminary_maintenance_suggestion,
    pick_highest_severity,
)

MODEL_PATH = Path("models/best.pt")

try:
    import torch
except ImportError:  # pragma: no cover - depends on environment
    torch = None

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - depends on environment
    YOLO = None


@dataclass
class Detection:
    raw_label: str
    display_label: str
    bbox: tuple[int, int, int, int]
    score: float
    mask: np.ndarray | None
    p25_width_px: float | None
    p25_width_mm: float | None
    severity: str
    width_message: str
    maintenance_suggestion: str


@dataclass
class InferenceSummary:
    issue_count: int
    main_defect_type: str
    top_predicted_class: str
    confidence_score: float
    detections: list[Detection]
    annotated_image: Image.Image
    message: str
    representative_p25_width_px: float | None
    representative_p25_width_mm: float | None
    auto_severity: str
    preliminary_maintenance_suggestion: str
    width_summary_message: str
    mm_per_px: float


_MODEL: YOLO | None = None
_PALETTE = ["#D62828", "#F77F00", "#588157", "#1D4ED8", "#6A4C93"]


def normalize_display_label(label: str) -> str:
    normalized = label.strip()
    if "crack" in normalized.lower():
        return "Crack"
    return normalized


def _select_device() -> str:
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_path: Path = MODEL_PATH) -> YOLO:
    global _MODEL

    if YOLO is None:
        raise RuntimeError("Ultralytics가 설치되어 있지 않습니다. requirements.txt를 설치하세요.")
    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    if _MODEL is None:
        _MODEL = YOLO(str(model_path))

    return _MODEL


def _empty_summary(image: Image.Image, message: str, mm_per_px: float) -> InferenceSummary:
    return InferenceSummary(
        issue_count=0,
        main_defect_type="No defect detected",
        top_predicted_class="None",
        confidence_score=0.0,
        detections=[],
        annotated_image=image.copy(),
        message=message,
        representative_p25_width_px=None,
        representative_p25_width_mm=None,
        auto_severity="Unavailable",
        preliminary_maintenance_suggestion="예비 판단으로는 균열 폭 정보가 없어 현장 확인이 필요합니다.",
        width_summary_message="균열 폭 측정 결과가 없습니다.",
        mm_per_px=mm_per_px,
    )


def _run_model_prediction(model: YOLO, image: Image.Image):
    image_array = np.array(image)
    preferred_device = _select_device()

    try:
        results = model.predict(image_array, device=preferred_device, verbose=False)
    except Exception:
        if preferred_device == "cpu":
            raise
        results = model.predict(image_array, device="cpu", verbose=False)

    if not results:
        return None

    return results[0]


def _extract_masks(result) -> list[np.ndarray | None]:
    masks = getattr(result, "masks", None)
    if masks is None or getattr(masks, "data", None) is None:
        box_count = len(result.boxes) if getattr(result, "boxes", None) is not None else 0
        return [None] * box_count

    mask_data = masks.data
    mask_array = mask_data.cpu().numpy() if hasattr(mask_data, "cpu") else np.asarray(mask_data)
    return [mask.astype(bool) for mask in mask_array]


def _build_detections(result, mm_per_px: float) -> list[Detection]:
    boxes = getattr(result, "boxes", None)
    names = getattr(result, "names", {}) or {}

    if boxes is None or boxes.cls is None or len(boxes) == 0:
        return []

    mask_values = _extract_masks(result)
    xyxy_values = boxes.xyxy.int().cpu().tolist()
    cls_values = boxes.cls.int().cpu().tolist()
    conf_values = boxes.conf.cpu().tolist() if boxes.conf is not None else []
    detections: list[Detection] = []

    for index, bbox in enumerate(xyxy_values):
        raw_label = names.get(cls_values[index], str(cls_values[index]))
        display_label = normalize_display_label(raw_label)
        score = float(conf_values[index]) if index < len(conf_values) else 0.0
        mask = mask_values[index] if index < len(mask_values) else None

        if display_label == "Crack" and mask is not None:
            width_measurement = estimate_crack_width(mask, mm_per_px=mm_per_px)
            severity = width_measurement.severity
            width_message = width_measurement.message
            p25_width_px = width_measurement.p25_width_px
            p25_width_mm = width_measurement.p25_width_mm
        elif display_label == "Crack":
            severity = "Unavailable"
            width_message = "세그멘테이션 마스크가 없어 균열 폭을 계산하지 못했습니다."
            p25_width_px = None
            p25_width_mm = None
        else:
            severity = "Not applicable"
            width_message = "해당 결함에는 균열 폭 기준을 직접 적용하지 않았습니다."
            p25_width_px = None
            p25_width_mm = None

        detections.append(
            Detection(
                raw_label=raw_label,
                display_label=display_label,
                bbox=(bbox[0], bbox[1], bbox[2], bbox[3]),
                score=score,
                mask=mask,
                p25_width_px=p25_width_px,
                p25_width_mm=p25_width_mm,
                severity=severity,
                width_message=width_message,
                maintenance_suggestion=get_preliminary_maintenance_suggestion(severity),
            )
        )

    return detections


def _summarize_detections(detections: list[Detection]) -> tuple[str, str, float]:
    display_labels = [detection.display_label for detection in detections]
    class_counts = Counter(display_labels)
    top_detection = max(detections, key=lambda detection: detection.score)
    most_common_count = max(class_counts.values())
    candidate_labels = {label for label, count in class_counts.items() if count == most_common_count}
    main_defect_type = (
        top_detection.display_label
        if top_detection.display_label in candidate_labels
        else next(iter(candidate_labels))
    )
    return main_defect_type, top_detection.display_label, top_detection.score


def _select_representative_crack(detections: list[Detection]) -> Detection | None:
    crack_detections = [
        detection for detection in detections if detection.display_label == "Crack" and detection.p25_width_mm is not None
    ]
    if not crack_detections:
        return None
    return max(crack_detections, key=lambda detection: (detection.p25_width_mm or 0.0, detection.score))


def _build_width_summary(
    detections: list[Detection],
) -> tuple[float | None, float | None, str, str, str]:
    crack_detections = [detection for detection in detections if detection.display_label == "Crack"]
    representative_crack = _select_representative_crack(detections)

    if not crack_detections:
        return (
            None,
            None,
            "Not applicable",
            get_preliminary_maintenance_suggestion("Not applicable"),
            "균열 탐지가 없어 폭 측정을 적용하지 않았습니다.",
        )

    if representative_crack is None:
        return (
            None,
            None,
            "Unavailable",
            get_preliminary_maintenance_suggestion("Unavailable"),
            crack_detections[0].width_message,
        )

    highest_severity = pick_highest_severity([detection.severity for detection in crack_detections])
    return (
        representative_crack.p25_width_px,
        representative_crack.p25_width_mm,
        highest_severity,
        get_preliminary_maintenance_suggestion(highest_severity),
        representative_crack.width_message,
    )


def _color_for_label(label: str) -> tuple[int, int, int]:
    if label == "Crack":
        return ImageColor.getrgb("#D62828")
    return ImageColor.getrgb(_PALETTE[hash(label) % len(_PALETTE)])


def _prepare_mask_layer(mask: np.ndarray, color: tuple[int, int, int], image_size: tuple[int, int]) -> Image.Image:
    mask_array = np.asarray(mask)
    if mask_array.ndim == 3:
        mask_array = mask_array[..., 0]
    mask_array = mask_array.astype(bool)
    mask_image = Image.fromarray(mask_array.astype(np.uint8) * 255, mode="L")
    if mask_image.size != image_size:
        mask_image = mask_image.resize(image_size, resample=Image.Resampling.NEAREST)
    mask_layer = Image.new("RGBA", image_size, (0, 0, 0, 0)).convert("RGBA")
    alpha_mask = mask_image.point(lambda value: 90 if value > 0 else 0)
    mask_layer.paste((*color, 255), (0, 0, image_size[0], image_size[1]), alpha_mask)
    return mask_layer


def _render_annotated_image(image: Image.Image, detections: list[Detection]) -> Image.Image:
    annotated = image.convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0)).convert("RGBA")
    image_width, image_height = annotated.size

    for detection in detections:
        if detection.mask is None:
            continue
        color = _color_for_label(detection.display_label)
        mask_layer = _prepare_mask_layer(detection.mask, color, annotated.size).convert("RGBA")
        if mask_layer.size != overlay.size:
            mask_layer = mask_layer.resize(overlay.size, resample=Image.Resampling.NEAREST)
        overlay = Image.alpha_composite(overlay, mask_layer)

    annotated = Image.alpha_composite(annotated, overlay)
    draw = ImageDraw.Draw(annotated)

    for detection in detections:
        color = _color_for_label(detection.display_label)
        left, top, right, bottom = detection.bbox
        left = max(0, min(int(left), image_width - 1))
        right = max(0, min(int(right), image_width - 1))
        top = max(0, min(int(top), image_height - 1))
        bottom = max(0, min(int(bottom), image_height - 1))
        draw.rectangle((left, top, right, bottom), outline=color, width=3)

        label_parts = [detection.display_label, f"{detection.score:.2f}"]
        if detection.p25_width_mm is not None:
            label_parts.append(f"{detection.p25_width_mm:.2f} mm")
        label_text = " | ".join(label_parts)

        text_top = max(top - 24, 0)
        text_right = min(image_width - 1, left + 220)
        draw.rectangle((left, text_top, text_right, top), fill=color)
        draw.text((left + 6, text_top + 4), label_text, fill="white")

    return annotated.convert("RGB")


def run_inference(image: Image.Image, mm_per_px: float = MM_PER_PX) -> InferenceSummary:
    try:
        model = load_model()
    except Exception as error:
        return _empty_summary(image, f"모델 로딩에 실패했습니다: {error}", mm_per_px)

    try:
        result = _run_model_prediction(model, image)
    except Exception as error:
        return _empty_summary(image, f"추론에 실패했습니다: {error}", mm_per_px)

    if result is None:
        return _empty_summary(image, "결함이 탐지되지 않았습니다.", mm_per_px)

    detections = _build_detections(result, mm_per_px=mm_per_px)
    if not detections:
        return _empty_summary(image, "결함이 탐지되지 않았습니다.", mm_per_px)

    main_defect_type, top_predicted_class, confidence_score = _summarize_detections(detections)
    annotated_image = _render_annotated_image(image, detections)
    representative_p25_width_px, representative_p25_width_mm, auto_severity, maintenance_suggestion, width_summary_message = _build_width_summary(detections)

    return InferenceSummary(
        issue_count=len(detections),
        main_defect_type=main_defect_type,
        top_predicted_class=top_predicted_class,
        confidence_score=confidence_score,
        detections=detections,
        annotated_image=annotated_image,
        message="결함 탐지와 폭 측정을 완료했습니다.",
        representative_p25_width_px=representative_p25_width_px,
        representative_p25_width_mm=representative_p25_width_mm,
        auto_severity=auto_severity,
        preliminary_maintenance_suggestion=maintenance_suggestion,
        width_summary_message=width_summary_message,
        mm_per_px=mm_per_px,
    )
