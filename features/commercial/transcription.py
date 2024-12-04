import json 
import PIL.Image
import typing_extensions as typing
from features.utils import read_file, convert_video_to_audio, is_supported_audio, is_supported_video
from model import genai_model, genai
class Transcription(typing.TypedDict):
    transcript: str

def transcribe_audio(file_path: str):

    if not is_supported_audio(file_path):
        raise Exception("Un-Supported file type for audio transcription")
    
    file = read_file(file_path)

    result = genai_model.generate_content(
        [file, "Generate a transcript of the audio."], 
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=Transcription
        ),
        request_options={"timeout": 600}
    )

    if isinstance(file, genai.types.File):
        file.delete()

    return json.loads(result)["transript"]

def transcribe_video(file_path: str):

    if not is_supported_video(file_path):
        raise Exception("Un-Supported file type for video transcription")
        
    file_path = convert_video_to_audio(file_path)
    file = read_file(file_path)

    result = genai_model.generate_content(
        [file, "Generate a transcript of the audio."], 
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=Transcription
        ),
        request_options={"timeout": 600}
    )

    if isinstance(file, genai.types.File):
        file.delete()

    return json.loads(result)["summary"]