from pydantic import BaseModel
from typing import List, Optional, Dict

# Translation schemas
class TranslationRequest(BaseModel):
    text: str
    target_language: str
    source_language: str = "auto"

class TranslationResponse(BaseModel):
    translated_text: str

# Summarization schemas
class SummarizationResponse(BaseModel):
    summary: str

# Zero-shot classification schemas
class ClassificationRequest(BaseModel):
    labels: List[str]
    text: str
    hypothesis: Optional[str]

class ClassificationResponse(BaseModel):
    classification: Dict[str, float]

# Text-to-Speech schemas
class TTSRequest(BaseModel):
    text: str

class KeywordExtractionResponse(BaseModel):
    keywords: List[str]

# ASR schemas
class TranscriptionResponse(BaseModel):
    transcript: str

class ChatwFileResponse(BaseModel):
    response: str