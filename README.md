# 외벽 결함 점검 앱

YOLOv8 세그멘테이션, 중심선 기반 균열폭 측정, 스케일 보정 입력, 자동 등급 판정, CSV 저장, 한국어 PDF 보고서 생성을 지원하는 Streamlit 앱입니다.

## 설치

```bash
pip install -r requirements.txt
```

학습된 모델 파일을 `models/best.pt` 위치에 두세요.


## 데모 버전 실행 사이트

https://apartment-crack-inspection-app-bkvazrxbg6rnjgnwuwkhyf.streamlit.app/
(접속 상태에 따라 링크가 안열릴 수 있습니다.)

## 실행

```bash
streamlit run app.py
```

## 주요 기능

- 외벽 이미지 업로드 및 YOLOv8 세그멘테이션 추론
- 스케일 보정 입력: 고정 보정값, mm/px 직접 입력, 기준 길이 환산
- 대표 p25 균열폭 기반 자동 등급 산정
- 균열 상태와 균열 거동 입력 반영
- `utf-8-sig` 인코딩 CSV 저장
- 스케일 정보, 판단 근거, 참고문헌을 포함한 한국어 PDF 보고서 생성

## 설정 메모

- 기본 보정값은 [src/width_estimation.py](C:\Users\User\Documents\New project\src\width_estimation.py)의 `MM_PER_PX`에서 조정할 수 있습니다.
- PDF 참고문헌 목록과 선택적 실무 참고 URL은 [src/report.py](C:\Users\User\Documents\New project\src\report.py)에서 수정할 수 있습니다.
