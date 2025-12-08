import os

# DPI settings for document conversion
CONVERSION_DPI = 300

# YOLO model configuration
YOLO_MODEL_PATH = os.path.join("model", "doclayout_yolo_docstructbench_imgsz1280_2501.pt")

# Data configuration
DATA_CSV_PATH = os.path.join("data", "test.csv")

# OCR model configuration
OCR_DET_MODEL_DIR = "model/PP-OCRv5_server_det"
OCR_REC_MODEL_DIR = "model/korean_PP-OCRv5_mobile_rec"

# Prediction configuration
PREDICTION_CONFIDENCE_THRESHOLD = 0.1
PREDICTION_INPUT_SIZE = 1600

# Device configuration
DEVICE = "cuda:0"

# Output configuration
OUTPUT_PATH = "output/submission.csv"
TEMP_DIR = "./temp_images"