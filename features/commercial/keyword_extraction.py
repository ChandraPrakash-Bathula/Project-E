from features.utils import read_file
from model import genai_model, genai
import typing_extensions as typing
import json

class KeywordOutput(typing.TypedDict):
    keywords = list[str]

def extract_keywords(file_path):
    file = read_file(file_path)

    result = genai_model.generate_content(
        [file, "Please give me a list of important keywords which are in here"], 
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=KeywordOutput
        ),
        request_options={"timeout": 600}
    )

    if isinstance(file, genai.types.File):
        file.delete()

    return json.loads(result)["keywords"]