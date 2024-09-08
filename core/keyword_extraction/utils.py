import nltk
import nltk
from nltk.corpus import words
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt')
nltk.download("words")

def split_into_paragraphs(self, doc: str, tokenizer,  max_tokens_per_para: int = 128):
    sentences = sent_tokenize(doc.strip())
    temp = ""
    temp_list = []
    final_list = []

    for i, sentence in enumerate(sentences):
        sent = sentence
        temp = temp + " " + sent
        wc_temp = len(tokenizer.tokenize(temp))

        if wc_temp < max_tokens_per_para:
            temp_list.append(sentence)

            if i == len(sentences) - 1:
                final_list.append(" ".join(temp_list))

        else:
            final_list.append(" ".join(temp_list))

            temp = sentence
            temp_list = [sentence]

            if i == len(sentences) - 1:
                final_list.append(" ".join(temp_list))

    return [para for para in final_list if len(para.strip()) != 0]

def process_outputs(outputs):
    temp = [output[0].split(" | ") for output in outputs]
    flatten = [item for sublist in temp for item in sublist]
    return sorted(set(flatten), key=flatten.index)

def filter_outputs(key_phrases, text):
    key_phrases = [elem.lower() for elem in key_phrases]
    text = text.lower()

    valid_phrases = []
    invalid_phrases = []

    for phrases in key_phrases:
        for phrase in word_tokenize(phrases):
            if (phrase in word_tokenize(text)) or (phrase in words.words()):
                if phrases not in valid_phrases:
                    valid_phrases.append(phrases)
            else:
                invalid_phrases.append(phrases)

    return [elem for elem in valid_phrases if elem not in invalid_phrases]