from transformers import pipeline
from features.utils import get_current_device

pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=get_current_device(), cache_dir="./model") # crosslingual: joeddav/xlm-roberta-large-xnli
sentiment_labels = [
            "anxious",
            "worried",
            "confused",
            "neutral",
            "hopeful",
            "relieved",
            "confident",
            "frustrated",
            "overwhelmed",
            "determined",
            "extreme"
        ]

def classify_text(text, labels, hypothesis=None):
    if hypothesis:
        result = pipe(text, labels, hypothesis_template=hypothesis, multi_label=False)
    else:
        result = pipe(text, labels, multi_label=False)
    classification_result = {k:v for k, v in zip(result['labels'], result["scores"])}
    return classification_result