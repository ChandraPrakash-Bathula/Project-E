import json 
import PIL.Image
import typing_extensions as typing
from features.utils import read_file
from model import genai_model, genai


class SummaryOutput(typing.TypedDict):
    summary: str

def summarize(file_path: str):

    file = read_file(file_path)

    result = genai_model.generate_content(
        [file, "Please summarize this file"], 
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=SummaryOutput
        ),
        request_options={"timeout": 600}
    )

    if isinstance(file, genai.types.File):
        file.delete()

    return json.loads(result)["summary"]