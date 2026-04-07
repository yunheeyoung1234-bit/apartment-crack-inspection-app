from datetime import datetime

import streamlit as st
from PIL import Image

from src.inference import run_inference
from src.report import generate_pdf_report
from src.storage import save_inspection_record
from src.width_estimation import MM_PER_PX


st.set_page_config(
    page_title="외벽 결함 점검",
    page_icon="🏢",
    layout="centered",
)


def main() -> None:
    st.title("외벽 결함 점검")
    st.caption("AI 탐지와 균열폭 기반 자동 판정을 바탕으로 예비 점검 결과를 정리합니다.")

    uploaded_file = st.file_uploader(
        "점검 이미지 업로드",
        type=["png", "jpg", "jpeg"],
        help="지원 형식: PNG, JPG, JPEG",
    )

    st.markdown("### 기본정보")
    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        facility_name = st.text_input("대상시설", value="공동주택 외벽")
        inspection_period = st.text_input("점검시기", value="정기점검")
    with info_col2:
        inspection_type = st.text_input("점검구분", value="예비점검")
        inspection_location = st.text_input("점검위치", value="외벽 전면")
    with info_col3:
        inspection_date = st.text_input("점검일자", value=datetime.now().strftime("%Y-%m-%d"))
        inspector_name = st.text_input("점검자", value="담당자")

    st.markdown("### 스케일 보정")
    scale_mode = st.selectbox(
        "보정 방식",
        options=["고정 보정값 사용", "mm/px 직접 입력", "기준 길이로 계산"],
        index=0,
    )

    mm_per_px, scale_error = resolve_mm_per_px(scale_mode)
    if scale_mode == "고정 보정값 사용":
        st.caption(f"기본 보정값 {MM_PER_PX:.6f} mm/px 를 사용합니다.")
    elif scale_error is None:
        st.caption(f"적용 예정 보정값: {mm_per_px:.6f} mm/px")

    col_state, col_movement = st.columns(2)
    with col_state:
        crack_condition = st.selectbox("균열 상태", options=["건조", "습윤", "누수"], index=0)
    with col_movement:
        crack_movement = st.selectbox("균열 거동", options=["정지 균열", "진행 의심"], index=0)

    inspector_comment = st.text_area(
        "점검 의견",
        placeholder="현장 관찰 사항, 보수 검토 메모, 추적 점검 계획 등을 입력합니다.",
        height=120,
    )

    if uploaded_file is None:
        st.info("이미지를 업로드하면 점검 결과가 표시됩니다.")
        return

    if scale_error is not None:
        st.error(scale_error)
        return

    original_image = Image.open(uploaded_file).convert("RGB")
    summary = run_inference(original_image, mm_per_px=mm_per_px)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("원본 이미지")
        st.image(original_image, width="stretch")
    with col2:
        st.subheader("주석 결과")
        st.image(summary.annotated_image, width="stretch")

    st.markdown("### 점검 요약")
    if summary.issue_count == 0:
        st.warning(summary.message)
    else:
        st.success(summary.message)

    st.write(f"탐지 개수: {summary.issue_count}")
    st.write(f"대표 결함 유형: {summary.main_defect_type}")
    st.write(f"최상위 예측 클래스: {summary.top_predicted_class}")
    st.write(f"대표 신뢰도: {summary.confidence_score:.2f}")
    st.write(f"대표 p25 균열폭(px): {format_optional_number(summary.representative_p25_width_px)}")
    st.write(f"대표 p25 균열폭(mm): {format_optional_number(summary.representative_p25_width_mm)}")
    st.write(f"자동 판정 등급: {summary.auto_severity}")
    st.write(f"스케일입력방식: {scale_mode}")
    st.write(f"적용 mm_per_px: {summary.mm_per_px:.6f}")
    st.write(f"균열 상태 입력: {crack_condition}")
    st.write(f"균열 거동 입력: {crack_movement}")
    st.write(f"폭 측정 메모: {summary.width_summary_message}")
    st.write("예비 보수 제안:")
    st.write(summary.preliminary_maintenance_suggestion)
    st.write("점검 의견:")
    st.write(inspector_comment or "입력된 점검 의견 없음.")

    if summary.detections:
        st.markdown("### 탐지 상세")
        st.dataframe(
            [
                {
                    "결함 라벨": detection.display_label,
                    "신뢰도": round(detection.score, 4),
                    "p25 폭(px)": format_optional_number(detection.p25_width_px),
                    "p25 폭(mm)": format_optional_number(detection.p25_width_mm),
                    "자동 등급": detection.severity,
                }
                for detection in summary.detections
            ],
            hide_index=True,
            width="stretch",
        )

    col_save, col_pdf = st.columns(2)
    with col_save:
        if st.button("CSV 저장", type="primary", width="stretch"):
            timestamp = datetime.now().isoformat(timespec="seconds")
            try:
                csv_path = save_inspection_record(
                    filename=uploaded_file.name,
                    timestamp=timestamp,
                    detected_issue_count=summary.issue_count,
                    predicted_defect_type=summary.main_defect_type,
                    top_predicted_class=summary.top_predicted_class,
                    confidence_score=summary.confidence_score,
                    p25_width_px=summary.representative_p25_width_px,
                    p25_width_mm=summary.representative_p25_width_mm,
                    auto_severity=summary.auto_severity,
                    scale_input_method=scale_mode,
                    applied_mm_per_px=summary.mm_per_px,
                    inspector_comment=inspector_comment,
                )
                st.success(f"점검 기록을 저장했습니다: {csv_path}")
            except Exception as error:
                st.error(f"CSV 저장 중 오류가 발생했습니다: {error}")

    with col_pdf:
        if st.button("PDF 생성 준비", width="stretch"):
            timestamp = datetime.now().isoformat(timespec="seconds")
            try:
                pdf_bytes, download_name = generate_pdf_report(
                    filename=uploaded_file.name,
                    predicted_defect_type=summary.main_defect_type,
                    detected_issue_count=summary.issue_count,
                    confidence_score=summary.confidence_score,
                    p25_width_px=summary.representative_p25_width_px,
                    p25_width_mm=summary.representative_p25_width_mm,
                    auto_severity=summary.auto_severity,
                    scale_input_method=scale_mode,
                    applied_mm_per_px=summary.mm_per_px,
                    inspector_comment=inspector_comment,
                    preliminary_maintenance_suggestion=summary.preliminary_maintenance_suggestion,
                    width_summary_message=summary.width_summary_message,
                    crack_condition=crack_condition,
                    crack_movement=crack_movement,
                    original_image=original_image,
                    annotated_image=summary.annotated_image,
                    facility_name=facility_name,
                    inspection_type=inspection_type,
                    inspection_period=inspection_period,
                    inspection_date=inspection_date,
                    inspection_location=inspection_location,
                    inspector_name=inspector_name,
                    timestamp=timestamp,
                )
                st.session_state["pdf_bytes"] = pdf_bytes
                st.session_state["pdf_download_name"] = download_name
                st.success("PDF가 생성되었습니다. 아래 버튼으로 다운로드하세요.")
            except Exception as error:
                st.error(f"PDF 생성 중 오류가 발생했습니다: {error}")

    pdf_bytes = st.session_state.get("pdf_bytes")
    pdf_download_name = st.session_state.get("pdf_download_name")
    if pdf_bytes and pdf_download_name:
        st.download_button(
            "PDF 다운로드",
            data=pdf_bytes,
            file_name=pdf_download_name,
            mime="application/pdf",
            width="stretch",
        )


def resolve_mm_per_px(scale_mode: str) -> tuple[float, str | None]:
    if scale_mode == "고정 보정값 사용":
        return MM_PER_PX, None

    if scale_mode == "mm/px 직접 입력":
        mm_per_px = st.number_input("mm_per_px", min_value=0.0, value=MM_PER_PX, step=0.001, format="%.6f")
        if mm_per_px <= 0:
            return 0.0, "mm/px 직접 입력 값은 0보다 커야 합니다."
        return float(mm_per_px), None

    actual_length_mm = st.number_input("기준 실제 길이(mm)", min_value=0.0, value=100.0, step=1.0, format="%.3f")
    pixel_length_px = st.number_input("기준 픽셀 길이(px)", min_value=0.0, value=500.0, step=1.0, format="%.3f")
    if actual_length_mm <= 0 or pixel_length_px <= 0:
        return 0.0, "기준 실제 길이와 기준 픽셀 길이는 모두 0보다 커야 합니다."
    return float(actual_length_mm / pixel_length_px), None


def format_optional_number(value: float | None) -> str:
    return "측정 불가" if value is None else f"{value:.2f}"


if __name__ == "__main__":
    main()
