from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MM_PER_PX = 0.20

SEVERITY_ORDER = {
    "Unavailable": -1,
    "Not applicable": 0,
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Critical": 4,
}

PRELIMINARY_MAINTENANCE_GUIDANCE = {
    "Low": "예비 판단으로는 추적 관찰 및 재점검을 우선 권장합니다.",
    "Medium": "예비 판단으로는 표면 실링 또는 국부 보수 검토가 필요합니다.",
    "High": "예비 판단으로는 상세 상태 검토와 균열 보수 계획 수립이 필요합니다.",
    "Critical": "예비 판단으로는 긴급 상세점검과 구조 검토를 우선 권장합니다.",
    "Unavailable": "예비 판단으로는 폭 측정 결과 확인이 어려워 현장 재검토가 필요합니다.",
    "Not applicable": "예비 판단으로는 해당 결함에 균열 폭 기준을 직접 적용하기 어렵습니다.",
}

try:
    from scipy.ndimage import distance_transform_edt
except ImportError:  # pragma: no cover - depends on environment
    distance_transform_edt = None

try:
    from skimage.morphology import skeletonize
except ImportError:  # pragma: no cover - depends on environment
    skeletonize = None


@dataclass
class WidthMeasurement:
    p25_width_px: float | None
    p25_width_mm: float | None
    severity: str
    message: str
    sample_count: int


def convert_px_to_mm(width_px: float | None, mm_per_px: float) -> float | None:
    if width_px is None:
        return None
    return width_px * mm_per_px


def assign_severity_from_width(p25_width_mm: float | None) -> str:
    if p25_width_mm is None:
        return "Unavailable"
    if p25_width_mm < 0.5:
        return "Low"
    if p25_width_mm < 1.0:
        return "Medium"
    if p25_width_mm < 2.0:
        return "High"
    return "Critical"


def get_preliminary_maintenance_suggestion(severity: str) -> str:
    return PRELIMINARY_MAINTENANCE_GUIDANCE.get(
        severity,
        "예비 판단으로는 추가 검토가 필요합니다.",
    )


def pick_highest_severity(severities: list[str]) -> str:
    if not severities:
        return "Unavailable"
    return max(severities, key=lambda severity: SEVERITY_ORDER.get(severity, -1))


def estimate_crack_width(mask: np.ndarray, mm_per_px: float = MM_PER_PX) -> WidthMeasurement:
    if skeletonize is None or distance_transform_edt is None:
        return WidthMeasurement(
            p25_width_px=None,
            p25_width_mm=None,
            severity="Unavailable",
            message="폭 추정 의존성이 없어 계산할 수 없습니다. requirements.txt를 확인하세요.",
            sample_count=0,
        )

    binary_mask = np.asarray(mask, dtype=bool)
    if binary_mask.ndim != 2 or binary_mask.sum() < 10:
        return WidthMeasurement(
            p25_width_px=None,
            p25_width_mm=None,
            severity="Unavailable",
            message="마스크가 너무 작아 신뢰성 있는 폭 추정이 어렵습니다.",
            sample_count=0,
        )

    centerline = skeletonize(binary_mask)
    centerline_points = np.argwhere(centerline)
    if len(centerline_points) < 2:
        return WidthMeasurement(
            p25_width_px=None,
            p25_width_mm=None,
            severity="Unavailable",
            message="중심선을 충분히 추출하지 못했습니다.",
            sample_count=0,
        )

    width_samples = _sample_widths_along_normals(binary_mask, centerline_points)
    message = "중심선 법선 방향 샘플링으로 균열 폭을 추정했습니다."

    if len(width_samples) < 3:
        width_samples = _distance_transform_widths(binary_mask, centerline)
        message = "중심선 거리변환 보조 방식으로 균열 폭을 추정했습니다."

    if len(width_samples) == 0:
        return WidthMeasurement(
            p25_width_px=None,
            p25_width_mm=None,
            severity="Unavailable",
            message="이 균열 마스크에서는 폭 추정에 실패했습니다.",
            sample_count=0,
        )

    p25_width_px = float(np.percentile(width_samples, 25))
    p25_width_mm = convert_px_to_mm(p25_width_px, mm_per_px)
    severity = assign_severity_from_width(p25_width_mm)

    return WidthMeasurement(
        p25_width_px=p25_width_px,
        p25_width_mm=p25_width_mm,
        severity=severity,
        message=message,
        sample_count=len(width_samples),
    )


def _sample_widths_along_normals(
    mask: np.ndarray,
    centerline_points: np.ndarray,
    neighborhood_radius: float = 4.0,
    step_size: float = 0.5,
) -> list[float]:
    height, width = mask.shape
    max_distance = float(np.hypot(height, width))
    points_xy = centerline_points[:, ::-1].astype(float)
    width_samples: list[float] = []

    for point in points_xy:
        tangent = _estimate_local_tangent(points_xy, point, radius=neighborhood_radius)
        if tangent is None:
            continue

        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        normal_norm = np.linalg.norm(normal)
        if normal_norm == 0:
            continue
        normal /= normal_norm

        positive_distance = _walk_until_boundary(mask, point, normal, max_distance, step_size)
        negative_distance = _walk_until_boundary(mask, point, -normal, max_distance, step_size)
        width_px = positive_distance + negative_distance
        if width_px > 0:
            width_samples.append(width_px)

    return width_samples


def _estimate_local_tangent(
    all_points_xy: np.ndarray,
    target_point_xy: np.ndarray,
    radius: float,
) -> np.ndarray | None:
    deltas = all_points_xy - target_point_xy
    distances = np.linalg.norm(deltas, axis=1)
    neighborhood = all_points_xy[(distances > 0) & (distances <= radius)]
    if len(neighborhood) < 2:
        return None

    centered = neighborhood - neighborhood.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    tangent = vh[0]
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm == 0:
        return None
    return tangent / tangent_norm


def _walk_until_boundary(
    mask: np.ndarray,
    start_xy: np.ndarray,
    direction_xy: np.ndarray,
    max_distance: float,
    step_size: float,
) -> float:
    last_inside_distance = 0.0
    current_distance = step_size

    while current_distance <= max_distance:
        x = start_xy[0] + direction_xy[0] * current_distance
        y = start_xy[1] + direction_xy[1] * current_distance
        ix = int(round(x))
        iy = int(round(y))

        if iy < 0 or ix < 0 or iy >= mask.shape[0] or ix >= mask.shape[1] or not mask[iy, ix]:
            break

        last_inside_distance = current_distance
        current_distance += step_size

    return last_inside_distance


def _distance_transform_widths(mask: np.ndarray, centerline: np.ndarray) -> list[float]:
    distance_map = distance_transform_edt(mask)
    return (2.0 * distance_map[centerline]).astype(float).tolist()
