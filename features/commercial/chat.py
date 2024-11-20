import json 
import typing_extensions as typing
from features.utils import read_file
from model import genai_model, genai

class Output(typing.TypedDict):
    answer: str

def chat_with_file(question, file_path):

    # Can ask a question about any given file: text,
    # video (https://ai.google.dev/gemini-api/docs/vision?lang=python#technical-details-video),
    # image (https://ai.google.dev/gemini-api/docs/vision?lang=python#technical-details-image),
    # audio (https://ai.google.dev/gemini-api/docs/audio?lang=python#supported-formats),
    # document (supported doc: https://ai.google.dev/gemini-api/docs/document-processing?lang=python#technical-details)
    
    # Examples:
    # https://ai.google.dev/gemini-api/docs/vision?lang=python#prompt-multiple
    # https://ai.google.dev/gemini-api/docs/vision?lang=python#prompt-video
    # https://ai.google.dev/gemini-api/docs/vision?lang=python#transcribe-video

    file = read_file(file_path)

    result = genai_model.generate_content(
        [file, question], 
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=Output
        ),
        request_options={"timeout": 600}
    )

    if isinstance(file, genai.types.File):
        file.delete()

    return json.loads(result)["answer"]