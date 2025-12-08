import cv2
from paddleocr import PaddleOCR
import numpy as np
from config import OCR_DET_MODEL_DIR, OCR_REC_MODEL_DIR

reader = PaddleOCR(
    det_model_dir=OCR_DET_MODEL_DIR,
    rec_model_dir=OCR_REC_MODEL_DIR,
    lang="korean",
    use_textline_orientation=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False
)

def run_ocr(image, bbox):
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return ""

    result = reader.predict(crop)
    texts = []
    for res in result:
        texts.extend(res['rec_texts'])

    return " ".join(texts)