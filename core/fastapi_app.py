from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import io
from features.summarization import with_slm, with_summarization_pipe
from features.asr import transcribe_video, transcribe_audio
from features.keyword_extraction import extract_keywords
from features.translation import translate_text
from features.tts import text2speech
from features.zsclassifier import classify_text
from features.VQA.main import chat_with_image

from schema import (
    TranslationRequest, TranslationResponse,
    SummarizationResponse, ClassificationRequest, ClassificationResponse, TTSRequest,
    KeywordExtractionResponse, TranscriptionResponse, ChatwImageRequest, ChatwImageResponse
)

app = FastAPI(title="EliteNotes API")

@app.post("/summarize/", response_model=SummarizationResponse)
async def summarization(file: UploadFile = File(...)):
    file_path = f"./uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    summary = with_slm(file_path)
    os.remove(file_path)
    
    return SummarizationResponse(summary=summary)

@app.post("/extract_keywords/", response_model=KeywordExtractionResponse)
async def keyword_extraction(file: UploadFile = File(...)):
    file_path = f"./uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    keywords = extract_keywords(file_path)    
    os.remove(file_path)
    
    return KeywordExtractionResponse(keywords=keywords)

@app.post("/transcribe/", response_model=TranscriptionResponse)
async def transcribe_file(file: UploadFile = File(...)):
    file_path = f"./uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    if file.filename.endswith(('.mp4', '.avi', '.mov')):
        transcript = transcribe_video(file_path)
    elif file.filename.endswith(('.mp3', '.wav', '.ogg')):
        transcript = transcribe_audio(file_path)
    else:
        raise ValueError("Unsupported file format")
    
    os.remove(file_path)
    
    return TranscriptionResponse(transcript=transcript)

@app.post("/translate/", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    translated = translate_text(request.text, request.target_language, request.source_language)
    return TranslationResponse(translated_text=translated)

@app.post("/text_to_speech/")
async def convert_text_to_speech(request: TTSRequest):
    audio_path = "./tts_output.wav"
    text2speech(request.text, audio_path)
    return FileResponse(audio_path, media_type="audio/wav", filename="tts_output.wav")

@app.post("/classify/")
async def classify(request: ClassificationRequest):
    result = classify_text(request.text, request.labels, request.hypothesis)
    return ClassificationResponse(classification=result)

@app.post("/chat_with_image/", response_model=ChatwImageResponse)
async def chat_with_image_route(file: UploadFile, text: str, target_language: Optional[str] = "en"):

    # Read the uploaded image file
    image_bytes = await file.read()
    image = io.BytesIO(image_bytes)
    
    # Call the chat_with_image function
    result = chat_with_image(image, text, target_language)
    return ChatwImageResponse(result=result)