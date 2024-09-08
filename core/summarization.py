from transformers import pipeline
import torch

# Initialize summarizer when the application starts
model_name = "facebook/bart-large-cnn"
summarizer = pipeline('summarization', model=model_name, torch_dtype=torch.float16, framework="pt", device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def summarize_text(text):
    summarized = summarizer(text, max_length=650, min_length=40, do_sample=True)
    summary_text = summarized[0]['summary_text']
    return summary_text
