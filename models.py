import os
import pandas as pd
from doclayout_yolo import YOLOv10
from config import YOLO_MODEL_PATH, DATA_CSV_PATH, DEVICE

def load_yolo_model():
    model = YOLOv10(model=YOLO_MODEL_PATH, task="detect", verbose=True)
    model.to(DEVICE)
    return model

def load_data():
    return pd.read_csv(DATA_CSV_PATH)