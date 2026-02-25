# 🛡 AI 번견 (AI CCTV Monitoring System)

YOLOv8 + MMAction2 기반 실시간 이상행동 감지 및 무인점포 보안 AI 시스템  

본 프로젝트는 CCTV 영상에서 도난 가능 행동을 자동 탐지하고,  
위험도를 분석하여 자동 녹화 및 알림을 수행하는 딥러닝 기반 모니터링 시스템입니다.

---

# 📌 Project Overview

본 시스템은 다음 기능을 제공합니다:

- 🎥 실시간 CCTV 스트리밍 (최대 4채널)
- 🤖 YOLOv8 기반 사람 객체 탐지
- 🧠 MMAction2 기반 행동 분석 (도난 / 구매 구분)
- 📊 확률 기반 위험도 자동 분류
- 🎬 이벤트 발생 시 자동 10초 녹화
- 🔔 카카오톡 및 경고음 알림 전송
- 💾 MongoDB GridFS 기반 영상 저장 및 관리
- 📈 위험도 통계 시각화 대시보드 제공

---

# 🧠 AI Model Architecture

본 프로젝트는 단일 모델이 아닌 **2단계 추론 구조**로 설계되었습니다.

---

## 1️⃣ Human Detection Model (YOLOv8)

- Framework: Ultralytics YOLOv8
- Task: 프레임 단위 사람 객체 탐지
- 실시간 스트리밍 환경에서 저지연 추론

### 🔧 적용 전략

- Confidence Threshold 튜닝
- IoU Threshold 최적화
- False Positive 감소 필터링
- 사람 탐지 시에만 행동 분석 단계 진입

---

## 2️⃣ Behavior Recognition Model (MMAction2)

- Framework: MMAction2 (TSM 기반)
- Task: 8프레임 시퀀스 행동 인식
- 도난 행동 / 구매 행동 확률 산출

---

# 📂 Dataset

본 프로젝트는 AI-Hub 공식 데이터를 기반으로 학습을 진행했습니다.

## 🛒 구매 행동 데이터

AI-Hub 편의점 구매 행동 데이터셋  
https://aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=%ED%8E%B8%EC%9D%98%EC%A0%90&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=71549  

- 정상 구매 행동 영상 데이터
- 계산대 이용, 상품 집기 후 결제 등 다양한 시나리오 포함
- 행동 인식 학습 데이터로 활용

---

## 🚨 도난 행동 데이터

AI-Hub 편의점 도난 행동 데이터셋  
https://aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=%ED%8E%B8%EC%9D%98%EC%A0%90&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=71550  

- 물품 은닉, 미결제 이탈 등 도난 시나리오 포함
- 구매 행동과의 구분 학습을 위해 활용

---

### 📊 데이터 전처리

- 영상 → 프레임 단위 분할
- 8프레임 시퀀스 구성
- 행동 라벨 재정의 (Purchase / Steal)
- 학습/검증 데이터 분리
- 클래스 불균형 완화를 위한 샘플링 전략 적용

---

# ⚙️ System Pipeline

Live Camera Input  
→ OpenCV Frame Capture  
→ YOLOv8 Human Detection  
→ (Human Detected)  
→ 8-Frame Sequence 구성  
→ MMAction2 행동 분석  
→ 위험도 평가  
→ 자동 10초 녹화  
→ MongoDB 저장  
→ 카카오톡 알림 전송  

---

# 📊 Risk Classification Logic

| 조건 | 위험도 |
|------|--------|
| 구매 확률 ≥ 0.80 | 안전 (Safety) |
| 물품습득 확률 ≥ 0.97 AND 구매 확률 < 0.80 | 위험 (Danger) |
| 물품습득 확률 ≥ 0.90 AND 구매 확률 < 0.80 | 주의 (Warning) |
| 그 외 | 안전 |

단일 프레임 판단이 아닌  
**행동 확률 기반 분석 방식**으로 설계하여  
오탐지를 감소시키고 실제 상황 반영도를 향상시켰습니다.

---

# 🏗 System Architecture

Web Browser  
→ Flask Server (app.py)  
→ YOLOv8 (Human Detection)  
→ MMAction2 (Behavior Recognition)  
→ OpenCV (영상 처리)  
→ MongoDB (GridFS 영상 저장)  
→ Kakao API (알림 전송)

---

# 📁 Project Structure

```bash
AI_CCTV_System/
├── app.py
├── kakao_api.py
├── README.md
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── main.html
│   ├── recodingvideos.html
│   ├── statistics.html
│   └── error.html
├── static/
│   ├── css/
│   ├── js/
│   ├── img/
│   ├── audio/
│   ├── videos/
│   └── save_to_data/
└── logs/
```

---

🧩 Core Algorithm Design

사람 탐지 시에만 행동 분석 수행하여 연산 최적화

8프레임 시퀀스 기반 행동 인식 적용

확률 기반 위험도 계산 로직 설계

이벤트 발생 시 자동 10초 녹화

안전 영상 24시간 후 자동 삭제 스케줄러 구현


---

# 👨‍💻 My Role in the Project

Team Lead & AI Model Development

본 프로젝트에서 팀원들의 역할을 조율하며 전체 시스템 개발을 총괄하였습니다.  
AI 모델 설계 및 학습을 주도하고, 실시간 추론 파이프라인을 구축하여  
AI 기반 보안 시스템이 실제로 동작할 수 있도록 통합하였습니다.

### 🔹 주요 수행 내용

- 팀원 역할 분담 및 개발 일정 조율
- AI-Hub 데이터 기반 구매/도난 행동 데이터 전처리
- YOLOv8 사람 탐지 모델 학습 및 최적화
- MMAction2 기반 행동 인식 모델 학습 및 튜닝
- 확률 기반 위험도 평가 알고리즘 설계
- 2단계 추론 구조(Detection → Action Recognition) 설계
- 실시간 추론 성능 최적화 및 임계값 조정

수행자 [김동언]
