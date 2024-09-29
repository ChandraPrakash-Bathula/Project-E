from googletrans import Translator
translator = Translator(service_urls=["translate.googleapis.com", "translate.googleapis.co.in"], user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64)')

def translate_text(text, target_language, source_language="auto"):
    translated_text = translator.translate(text, dest=target_language, src=source_language)
    return translated_text.text.strip()