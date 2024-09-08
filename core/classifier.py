from transformers import pipeline

pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0) # crosslingual: joeddav/xlm-roberta-large-xnli
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

"""
I couldn't clearly figure out where we could incoporate this feature and how will the user benifit, it could be used to classify the text into predefined labels (labels can be sentiment, topic, intent ...)
For Example, (created using chatgpt)

Scenario:

Imagine a company that offers customer support for a range of products. They receive various types of inquiries and need to categorize these based on predefined labels to streamline their process.
Predefined Labels:

    Billing Issue
    Technical Support
    Product Information
    Order Status
    Feedback

Example Emails Input text:

    Email: "I haven't received my order yet, and it's been over two weeks. Can you provide an update?"
        Classification: Order Status

    Email: "The product I purchased stopped working after a week. I need help fixing it."
        Classification: Technical Support

    Email: "I'm interested in knowing more about the features of your latest product."
        Classification: Product Information

    Email: "I'm not happy with the extra charges on my bill this month. Please explain."
        Classification: Billing Issue

    Email: "Great job with the latest update! The new features are really helpful."
        Classification: Feedback
"""

def classify_text(self, text):
    result = self.pipeline(text, self.sentiment_labels)
    classification_result = result['labels'][0][0]
    return classification_result