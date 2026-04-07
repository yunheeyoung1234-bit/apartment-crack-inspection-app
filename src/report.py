from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path

from PIL import Image as PILImage

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
except ImportError:  # pragma: no cover - depends on environment
    A4 = None
    Image = None
    Paragraph = None
    ParagraphStyle = None
    SimpleDocTemplate = None
    Spacer = None
    TTFont = None
    Table = None
    TableStyle = None
    colors = None
    getSampleStyleSheet = None
    landscape = None
    mm = None
    pdfmetrics = None


PDF_FONT_NAME = "Helvetica"
OPTIONAL_PRACTICAL_REFERENCE_URL: str | None = None

LEGAL_REFERENCES = [
    "공동주택관리법 시행령 제34조(공동주택의 안전점검)",
]

TECHNICAL_REFERENCES = [
    "대한주택공사 주택도시연구원, 콘크리트 구조물의 균열보수재료 및 공법의 선정방법 연구, 2002.",
    "오상근, 누수균열 보수를 위한 원인 분석과 보수 재료 및 공법 선정 방법 - ISO TR 16475 지침 이해, 건설감리, 2012.",
    "KS F 4925 관련 실무 참고 기준",
]

DISCLAIMER_TEXT = (
    "본 문서는 AI 기반 예비 점검 지원용 자동 생성 결과이며, 최종 판정 및 보수 공법 선정은 "
    "점검자 또는 전문기술자의 검토를 필요로 함."
)


def generate_maintenance_reasoning(
    p25_width_mm: float | None,
    auto_severity: str,
    crack_condition: str,
    crack_movement: str,
) -> list[str]:
    reasons: list[str] = []
    width_text = _format_optional_number(p25_width_mm)

    if (
        p25_width_mm is not None
        and auto_severity in {"Low", "Medium"}
        and crack_condition == "건조"
        and crack_movement == "정지 균열"
    ):
        reasons.append(
            f"건조·정지 균열로 입력되었고 대표 균열폭이 {width_text} mm로 비교적 작아 추적관찰 및 표면 보수 중심으로 예비 제안함."
        )
    if crack_condition in {"습윤", "누수"}:
        reasons.append(f"{crack_condition} 조건이 입력되어 단순 폭 기준보다 수분 조건을 우선 반영하였음.")
    if crack_movement == "진행 의심":
        reasons.append("균열 거동 가능성이 있어 경질 재료 단독 적용보다 거동 대응성 검토가 필요하다고 판단함.")
    if auto_severity == "Critical":
        reasons.append("대표 균열폭이 크므로 보수공법 확정보다 상세점검 및 구조적 검토를 우선 권고함.")
    elif auto_severity == "High":
        reasons.append("대표 균열폭이 보수 계획 검토가 필요한 수준으로 평가되어 상세 상태 확인과 공법 검토를 함께 권고함.")
    if not reasons:
        reasons.append(
            f"대표 균열폭은 {width_text} mm, 자동 판정 등급은 {auto_severity}로 평가되어 해당 수준에 맞춘 예비 보수 방향을 제안함."
        )
    return reasons[:3]


def generate_pdf_report(
    filename: str,
    predicted_defect_type: str,
    detected_issue_count: int,
    confidence_score: float,
    p25_width_px: float | None,
    p25_width_mm: float | None,
    auto_severity: str,
    scale_input_method: str,
    applied_mm_per_px: float,
    inspector_comment: str,
    preliminary_maintenance_suggestion: str,
    width_summary_message: str,
    crack_condition: str,
    crack_movement: str,
    original_image: PILImage.Image,
    annotated_image: PILImage.Image,
    facility_name: str,
    inspection_type: str,
    inspection_period: str,
    inspection_date: str,
    inspection_location: str,
    inspector_name: str,
    timestamp: str | None = None,
    output_dir: Path = Path("."),
) -> Path:
    if any(
        item is None
        for item in [A4, Table, TableStyle, SimpleDocTemplate, Paragraph, colors, landscape, mm, pdfmetrics, TTFont]
    ):
        raise RuntimeError("ReportLab이 설치되어 있지 않습니다. requirements.txt를 설치하세요.")

    font_name = _register_korean_font()
    styles = _build_styles(font_name)

    report_timestamp = timestamp or datetime.now().isoformat(timespec="seconds")
    safe_stem = Path(filename).stem or "inspection"
    safe_timestamp = report_timestamp.replace(":", "-")
    output_path = output_dir / f"{safe_stem}_inspection_report_{safe_timestamp}.pdf"

    reasoning_lines = generate_maintenance_reasoning(
        p25_width_mm=p25_width_mm,
        auto_severity=auto_severity,
        crack_condition=crack_condition,
        crack_movement=crack_movement,
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=landscape(A4),
        leftMargin=10 * mm,
        rightMargin=10 * mm,
        topMargin=10 * mm,
        bottomMargin=10 * mm,
    )

    elements = [
        Paragraph("공동주택 외벽 균열 예비점검 결과서", styles["title"]),
        Paragraph("AI 기반 예비 점검 지원 결과", styles["subtitle"]),
        Spacer(1, 4 * mm),
        _section_title("1. 기본정보", styles),
        _basic_info_table(
            facility_name=facility_name,
            inspection_type=inspection_type,
            inspection_period=inspection_period,
            inspection_date=inspection_date,
            inspection_location=inspection_location,
            inspector_name=inspector_name,
            crack_condition=crack_condition,
            crack_movement=crack_movement,
            styles=styles,
        ),
        Spacer(1, 3 * mm),
        _section_title("2. 탐지결과", styles),
        _detection_result_table(
            predicted_defect_type=predicted_defect_type,
            detected_issue_count=detected_issue_count,
            confidence_score=confidence_score,
            p25_width_px=p25_width_px,
            p25_width_mm=p25_width_mm,
            auto_severity=auto_severity,
            scale_input_method=scale_input_method,
            applied_mm_per_px=applied_mm_per_px,
            styles=styles,
        ),
        Spacer(1, 3 * mm),
        _section_title("3. 점검영상", styles),
        _image_table(original_image, annotated_image, styles),
        Spacer(1, 3 * mm),
        _section_title("4. 판정 및 조치", styles),
        _judgement_table(
            predicted_defect_type=predicted_defect_type,
            auto_severity=auto_severity,
            preliminary_maintenance_suggestion=preliminary_maintenance_suggestion,
            reasoning_lines=reasoning_lines,
            width_summary_message=width_summary_message,
            inspector_comment=inspector_comment,
            styles=styles,
        ),
        Spacer(1, 3 * mm),
        _section_title("5. 참고문헌 및 적용 근거", styles),
        _reference_table(styles),
    ]

    doc.build(elements)
    return output_path


def _build_styles(font_name: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName=font_name,
            fontSize=18,
            leading=22,
            alignment=1,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["Normal"],
            fontName=font_name,
            fontSize=10,
            leading=13,
            alignment=1,
            textColor=colors.HexColor("#555555"),
            spaceAfter=0,
        ),
        "section": ParagraphStyle(
            "section",
            parent=base["Normal"],
            fontName=font_name,
            fontSize=10,
            leading=12,
            spaceAfter=2,
        ),
        "cell": ParagraphStyle(
            "cell",
            parent=base["Normal"],
            fontName=font_name,
            fontSize=9,
            leading=12,
            wordWrap="CJK",
        ),
        "cell_bold": ParagraphStyle(
            "cell_bold",
            parent=base["Normal"],
            fontName=font_name,
            fontSize=9,
            leading=12,
            wordWrap="CJK",
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["Normal"],
            fontName=font_name,
            fontSize=8,
            leading=10,
            wordWrap="CJK",
        ),
    }


def _section_title(text: str, styles: dict[str, ParagraphStyle]) -> Paragraph:
    return Paragraph(f"<b>{text}</b>", styles["section"])


def _p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph((text or "-").replace("\n", "<br/>"), style)


def _table_style() -> TableStyle:
    return TableStyle(
        [
            ("BOX", (0, 0), (-1, -1), 1, colors.black),
            ("INNERGRID", (0, 0), (-1, -1), 0.7, colors.black),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
    )


def _basic_info_table(
    facility_name: str,
    inspection_type: str,
    inspection_period: str,
    inspection_date: str,
    inspection_location: str,
    inspector_name: str,
    crack_condition: str,
    crack_movement: str,
    styles: dict[str, ParagraphStyle],
) -> Table:
    data = [
        [_p("대상시설", styles["cell_bold"]), _p(facility_name, styles["cell"]), _p("점검구분", styles["cell_bold"]), _p(inspection_type, styles["cell"])],
        [_p("점검시기", styles["cell_bold"]), _p(inspection_period, styles["cell"]), _p("점검일자", styles["cell_bold"]), _p(inspection_date, styles["cell"])],
        [_p("점검위치", styles["cell_bold"]), _p(inspection_location, styles["cell"]), _p("점검자", styles["cell_bold"]), _p(inspector_name, styles["cell"])],
        [_p("균열상태", styles["cell_bold"]), _p(crack_condition, styles["cell"]), _p("균열거동", styles["cell_bold"]), _p(crack_movement, styles["cell"])],
    ]
    table = Table(data, colWidths=[24 * mm, 66 * mm, 24 * mm, 66 * mm], repeatRows=0)
    style = _table_style()
    style.add("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAEAEA"))
    style.add("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#EAEAEA"))
    table.setStyle(style)
    return table


def _detection_result_table(
    predicted_defect_type: str,
    detected_issue_count: int,
    confidence_score: float,
    p25_width_px: float | None,
    p25_width_mm: float | None,
    auto_severity: str,
    scale_input_method: str,
    applied_mm_per_px: float,
    styles: dict[str, ParagraphStyle],
) -> Table:
    headers = [
        "결함종류",
        "탐지개수",
        "최고신뢰도",
        "대표 p25 균열폭(px)",
        "대표 p25 균열폭(mm)",
        "자동심각도",
        "스케일 입력 방식",
        "적용 mm_per_px",
    ]
    values = [
        predicted_defect_type,
        str(detected_issue_count),
        f"{confidence_score:.2f}",
        _format_optional_number(p25_width_px),
        _format_optional_number(p25_width_mm),
        auto_severity,
        scale_input_method,
        f"{applied_mm_per_px:.6f}",
    ]
    data = [[_p(header, styles["cell_bold"]) for header in headers], [_p(value, styles["cell"]) for value in values]]
    table = Table(
        data,
        colWidths=[28 * mm, 20 * mm, 22 * mm, 34 * mm, 34 * mm, 22 * mm, 34 * mm, 28 * mm],
        repeatRows=1,
    )
    style = _table_style()
    style.add("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAEAEA"))
    style.add("ALIGN", (1, 0), (-1, -1), "CENTER")
    table.setStyle(style)
    return table


def _image_table(
    original_image: PILImage.Image,
    annotated_image: PILImage.Image,
    styles: dict[str, ParagraphStyle],
) -> Table:
    left_image = _pil_to_flowable(original_image, 115 * mm, 68 * mm)
    right_image = _pil_to_flowable(annotated_image, 115 * mm, 68 * mm)
    data = [
        [_p("원본 이미지", styles["cell_bold"]), _p("주석 이미지", styles["cell_bold"])],
        [left_image, right_image],
        [_p("입력 영상 기준", styles["small"]), _p("AI 탐지 및 주석 반영", styles["small"])],
    ]
    table = Table(data, colWidths=[125 * mm, 125 * mm], rowHeights=[10 * mm, 72 * mm, 8 * mm])
    style = _table_style()
    style.add("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAEAEA"))
    style.add("ALIGN", (0, 0), (-1, -1), "CENTER")
    table.setStyle(style)
    return table


def _judgement_table(
    predicted_defect_type: str,
    auto_severity: str,
    preliminary_maintenance_suggestion: str,
    reasoning_lines: list[str],
    width_summary_message: str,
    inspector_comment: str,
    styles: dict[str, ParagraphStyle],
) -> Table:
    rows = [
        ["판정요약", f"대표 결함은 {predicted_defect_type}, 자동 심각도는 {auto_severity}로 정리됨."],
        ["예비 보수 제안", preliminary_maintenance_suggestion],
        ["판단 근거", "<br/>".join(f"- {line}" for line in reasoning_lines)],
        ["특이사항", width_summary_message],
        ["검토의견", inspector_comment.strip() or "입력된 점검 의견 없음."],
    ]
    data = [[_p(label, styles["cell_bold"]), _p(value, styles["cell"]) ] for label, value in rows]
    table = Table(data, colWidths=[34 * mm, 216 * mm])
    style = _table_style()
    style.add("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAEAEA"))
    table.setStyle(style)
    return table


def _reference_table(styles: dict[str, ParagraphStyle]) -> Table:
    practical_reference = OPTIONAL_PRACTICAL_REFERENCE_URL or "사용자 제공 실무 참고자료 없음"
    rows = [
        ["법령/기준", "<br/>".join(f"- {item}" for item in LEGAL_REFERENCES)],
        ["연구/기술자료", "<br/>".join(f"- {item}" for item in TECHNICAL_REFERENCES)],
        ["실무 참고자료", practical_reference],
        ["유의사항", DISCLAIMER_TEXT],
    ]
    data = [[_p(label, styles["cell_bold"]), _p(value, styles["cell"])] for label, value in rows]
    table = Table(data, colWidths=[34 * mm, 216 * mm])
    style = _table_style()
    style.add("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAEAEA"))
    table.setStyle(style)
    return table


def _pil_to_flowable(image: PILImage.Image, max_width: float, max_height: float) -> Image:
    image_copy = image.copy()
    image_copy.thumbnail((max_width, max_height), PILImage.Resampling.LANCZOS)
    buffer = BytesIO()
    image_copy.save(buffer, format="PNG")
    buffer.seek(0)
    flowable = Image(buffer)
    flowable._restrictSize(max_width, max_height)
    return flowable


def _register_korean_font() -> str:
    global PDF_FONT_NAME
    if PDF_FONT_NAME != "Helvetica":
        return PDF_FONT_NAME

    font_candidates = [
        ("MalgunGothic", Path(r"C:\Windows\Fonts\malgun.ttf")),
        ("AppleGothic", Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf")),
        ("NanumGothic", Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")),
    ]
    for font_name, font_path in font_candidates:
        if font_path.exists():
            if font_name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
            PDF_FONT_NAME = font_name
            return PDF_FONT_NAME
    return PDF_FONT_NAME


def _format_optional_number(value: float | None) -> str:
    return "측정 불가" if value is None else f"{value:.2f}"
