# import
import os
import sys
import nltk
from transformers import AutoTokenizer, T5ForConditionalGeneration, MT5ForConditionalGeneration
from .utils import split_into_paragraphs, process_outputs, filter_outputs

model = MT5ForConditionalGeneration.from_pretrained("snrspeaks/KeyPhraseTransformer")
tokenizer = AutoTokenizer.from_pretrained("snrspeaks/KeyPhraseTransformer")  

def predict(doc: str):
    input_ids = tokenizer.encode(
        doc, return_tensors="pt", add_special_tokens=True
    )
    generated_ids = model.generate(
        input_ids=input_ids,
        num_beams=2,
        max_length=512,
        repetition_penalty=2.5,
        length_penalty=1,
        early_stopping=True,
        top_p=0.95,
        top_k=50,
        num_return_sequences=1,
    )
    preds = [
        tokenizer.decode(
            g, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        for g in generated_ids
    ]
    return preds

def extract_keywords(self, text: str, text_block_size: int = 64):
    results = []
    paras = split_into_paragraphs(
        doc=text, tokenizer=tokenizer, max_tokens_per_para=text_block_size
    )

    for para in paras:
        results.append(self.predict(para))

    key_phrases = filter_outputs(process_outputs(results), text)
    return key_phrases