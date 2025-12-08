import os
import subprocess
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
from config import CONVERSION_DPI

def convert_to_images(input_path, temp_dir, dpi=CONVERSION_DPI):
    ext = Path(input_path).suffix.lower()
    os.makedirs(temp_dir, exist_ok=True)

    if ext == ".pdf":
        return convert_from_path(input_path, dpi=dpi, output_folder=temp_dir, fmt="png")
    elif ext == ".pptx":
        subprocess.run([
            "libreoffice", "--headless", "--convert-to", "pdf", "--outdir", temp_dir, input_path
        ], check=True)
        pdf_path = os.path.join(temp_dir, Path(input_path).with_suffix(".pdf").name)
        return convert_from_path(pdf_path, dpi=dpi, output_folder=temp_dir, fmt="png")
    elif ext in [".jpg", ".jpeg", ".png"]:
        return [Image.open(input_path).convert("RGB")]
    else:
        raise ValueError(f"Unsupported file format: {ext}")