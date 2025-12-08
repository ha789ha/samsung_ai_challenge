import os
import pandas as pd
import cv2
import numpy as np
from PIL import Image
from converters import convert_to_images
from models import load_yolo_model, load_data
from ocr import run_ocr
from processing import clean_predictions, process_and_order_document
from utils import save_results_csv
from config import PREDICTION_CONFIDENCE_THRESHOLD, PREDICTION_INPUT_SIZE, DEVICE, OUTPUT_PATH, TEMP_DIR

def predict_with_ocr(yolo_model, data, input_size=PREDICTION_INPUT_SIZE, conf_threshold=PREDICTION_CONFIDENCE_THRESHOLD):
    all_results = []
    os.makedirs("temp", exist_ok=True)

    for _, row in data.iterrows():
        img_id = row["ID"]
        file_path = os.path.join("data", row["path"])
        full_id = img_id

        target_w = int(row.get("width", 1024))
        target_h = int(row.get("height", 1024))

        temp_image_dir = TEMP_DIR
        images = convert_to_images(file_path, temp_image_dir)
        img_pil = images[0]

        det_res = yolo_model.predict(img_pil, imgsz=input_size, conf=conf_threshold, device=DEVICE)
        boxes = det_res[0].boxes
        names = det_res[0].names

        img_cv2_original_ratio = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        orig_h, orig_w = img_cv2_original_ratio.shape[:2]

        for idx in range(len(boxes)):
            x1, y1, x2, y2 = map(int, (boxes.xyxy[idx].tolist()))
            conf = float(boxes.conf[idx].item())
            cls_idx = int(boxes.cls[idx].item())
            cls_name = names[cls_idx]

            if cls_name == "plain text": cls_name = "text"
            if cls_name == "figure": cls_name = "image"
            if cls_name == 'isolate_formula': cls_name = 'equation'

            extracted_text = ""
            if cls_name in {"title", "subtitle", "text"}:
                extracted_text = run_ocr(img_cv2_original_ratio, [int(x1), int(y1), int(x2), int(y2)])

            all_results.append({
                "ID": full_id,
                "category_type": cls_name,
                "confidence_score": round(conf, 3),
                "text": extracted_text,
                "bbox": f"{x1},{y1},{x2},{y2}"
            })

    df = pd.DataFrame(all_results)
    df = clean_predictions(df)
    return df

def main():
    yolo_model = load_yolo_model()
    data = load_data()
    predictions = predict_with_ocr(yolo_model, data)
    final_df = process_and_order_document(predictions)
    final_df.to_csv(OUTPUT_PATH, index=False)

if __name__=="__main__":
    main()
