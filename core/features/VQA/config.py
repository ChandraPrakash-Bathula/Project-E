from transformers import AutoModelForCausalLM, AutoProcessor
from ultralytics import YOLO
from googletrans import Translator

translator = Translator(service_urls=["translate.google.com", "translate.google.co.in"])

def load_language_model(model_path):
    processor = AutoProcessor.from_pretrained(model_path, torch_dtype='auto', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype='auto', device_map="auto", trust_remote_code=True)
    return processor, model

def load_yolo_model(model_path):
    return YOLO(model_path)

def translate_text(text, target_language, source_language="en"):
    try:
        translated_text = translator.translate(text, dest=target_language, src=source_language)
        return translated_text.text.strip()
    except Exception:
        return text

def detect_objects(image, yolo_model):
    results = yolo_model(image)
    return results[0].boxes.xyxy  # Bounding boxes in xyxy format

def show_image_with_boxes(image, boxes):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for box in boxes:
        x, y, x2, y2 = box[:4]
        width, height = x2 - x, y2 - y
        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()