from transformers import pipeline
from features.utils import extract_content_from_file
from features.utils import get_current_device

model_name = "facebook/bart-large-cnn"
summarizer = pipeline('summarization', model=model_name, framework="pt", device=get_current_device(), cache_dir="./model/")

def summarize_summarizartion_pipeline(file_path):
    text = extract_content_from_file(file_path)[:1024]
    summarized = summarizer(text, max_length=int(0.05 * len(str(text))), min_length=int(0.01 * len(str(text))), do_sample=False)
    summary_text = summarized[0]['summary_text']
    return summary_text