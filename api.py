from fastapi import FastAPI, UploadFile, File, Form
import os
from features.commercial.summarization import summarize
from features.commercial.transcription import transcribe_audio, transcribe_video
from features.commercial.keyword_extraction import extract_keywords
from features.commercial.translation import translate
from features.commercial.chat import chat_with_file
from features.utils import write_fastapi_file

from schema import (
    TranslationRequest, TranslationResponse,
    SummarizationResponse, ClassificationRequest, ClassificationResponse, TTSRequest,
    KeywordExtractionResponse, TranscriptionResponse, ChatwFileResponse
)

app = FastAPI(title="EliteNotes API")

@app.post("/summarize/", response_model=SummarizationResponse)
async def summarization(file: UploadFile = File(...)):
    
    file_path = write_fastapi_file(file)
    summary = summarize(file_path)
    os.remove(file_path)
    
    return SummarizationResponse(summary=summary)

@app.post("/extract_keywords/", response_model=KeywordExtractionResponse)
async def keyword_extraction(file: UploadFile = File(...)):
    
    file_path = write_fastapi_file(file)
    keywords = extract_keywords(file_path)    
    os.remove(file_path)
    
    return KeywordExtractionResponse(keywords=keywords)

@app.post("/transcribe_audio", response_model=TranscriptionResponse)
async def transcribe_file(file: UploadFile = File(...)):
    
    file_path = write_fastapi_file(file)
    transcript = transcribe_audio(file_path)
    os.remove(file_path)
    return TranscriptionResponse(transcript=transcript)

@app.post("/transcribe_video", response_model=TranscriptionResponse)
async def transcribe_file(file: UploadFile = File(...)):
    
    file_path = write_fastapi_file(file)
    transcript = transcribe_video(file_path)
    os.remove(file_path)
    return TranscriptionResponse(transcript=transcript)

@app.post("/translate/", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    translated = translate(request.text, request.target_language, request.source_language)
    return TranslationResponse(translated_text=translated)

# @app.post("/text_to_speech/")
# async def convert_text_to_speech(request: TTSRequest):
#     audio_path = "./tts_output.wav"
#     text2speech(request.text, audio_path)
#     return FileResponse(audio_path, media_type="audio/wav", filename="tts_output.wav")

@app.post("/chat_with_file", response_model=ChatwFileResponse)
async def chat_with_image_route(file: UploadFile = File(...), query: str = Form(...)):
    filepath = write_fastapi_file(file)
    result = chat_with_file(query, filepath)
    return ChatwFileResponse(result=result)