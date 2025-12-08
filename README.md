# Visually-rich Document Understanding  
### 2025 Samsung AI Challenge – 3rd Place
---

## 📌 Project Info
- **기간:** 2025.08.04 ~ 2025.09.12
- **대회링크:** https://dacon.io/competitions/official/236564/overview/description    
- **성과:** 3등  

---

## 📐 Pipeline
1. Layout Detection (DocLayout-YOLO)  
2. OCR (PP-OCRv5)  
3. 구조 분석 – Section → Column → Row 기반 정렬

---
## 💡 개발 주요사항 
1. 대회의 한정된 GPU 자원(T4)을 고려해 성능과 추론 속도를 모두 갖춘 YOLO 기반의 모델 활용
2. 사람이 읽는 순서를 반영한 후처리를 통해 문맥에 맞게 요소들을 정렬
---
## 🧑‍💻 My Role & Contributions
1. Layout Detection 개선을 위한 DocLayout-YOLO 기반 실험 및 파라미터 튜닝
2. OCR 파이프라인 구축 및 한국어 환경에 최적화 된 PP-OCRv5 버전 적용
3. 좌표를 바탕으로 정렬 순서 알고리즘 설계
---

## 📁 Output Examples

### 🖼️ Raw Document vs Detected Layout

| Raw Document | Detected Layout |
|--------------|------------------|
| <img src="https://github.com/user-attachments/assets/e8c51942-cfed-4955-8479-0d49dd215162" width="400"> | <img src="https://github.com/user-attachments/assets/c4365b60-dc2e-4953-8995-5d079e6662dd" width="400"> |


---

### 🏗️ Architecture Overview  

![architecture](https://github.com/user-attachments/assets/f08e9a50-bd21-4ef0-9407-c7083aaf78d7)


---

## 📦 Tech Stack
- **Detection:** DocLayout-YOLO  
- **OCR:** PP-OCRv5  
- **Language:** Python 3.10
- **Env:** NVIDIA T4  

---
## 코드 실행
### 1. Environment Setup
```bash
# 1. Repository Clone
git clone [https://github.com/ha789ha/samsung_ai_challenge.git](https://github.com/ha789ha/samsung_ai_challenge.git)
cd samsung_ai_challenge

# 2. Conda Environment Setup
conda create -n samsung python=3.10
conda activate samsung

# 3. Install Dependencies
pip install -r requirements.txt
```
### 2. Data & Model Preparation
- **Data (Test Images & CSV)**: [Dacon Data Download](https://dacon.io/competitions/official/236630/data)  
- **Detection Model**: [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOL) (`doclayout_yolo_docstructbench_imgsz1280_2501.pt`)  
- **OCR Model**: (PaddleOCR Model)[https://huggingface.co/PaddlePaddle/PP-OCRv5_server_det']  
  - Detection: `PP-OCRv5_server_det`  
  - Recognition: `korean_PP-OCRv5_mobile_rec`  
 
#### 📂 Directory Structure
```bash
samsung_ai_challenge/
├── data/
│ ├── test.csv
│ └── (images...) # Test 이미지 파일 위치
├── model/
│ ├── doclayout_yolo_docstructbench_imgsz1280_2501.pt
│ ├── PP-OCRv5_server_det/ # 압축 해제된 폴더
│ └── korean_PP-OCRv5_mobile_rec/ # 압축 해제된 폴더
├── output/ # 결과 저장 폴더
├── config.py
└── main.py # 실행 파일
```
### 3. Inference
```bash
python script.py
```

---
