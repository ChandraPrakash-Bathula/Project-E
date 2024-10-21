import os
from config import load_language_model, load_yolo_model, translate_text, detect_objects, show_image_with_boxes

LANGUAGE_MODEL_PATH = os.getenv("LANGUAGE_MODEL_PATH", "cyan2k/molmo-7B-D-bnb-4bit")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")

processor, model = load_language_model(LANGUAGE_MODEL_PATH)
yolo_model = load_yolo_model(YOLO_MODEL_PATH)
