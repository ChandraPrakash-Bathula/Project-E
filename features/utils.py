import docx
import requests
from bs4 import BeautifulSoup
import torch
from transformers import is_torch_npu_available
import google.generativeai as genai
import time
import pathlib
import PIL
from moviepy.video import VideoClip
from pydub import AudioSegment
import os
import mimetypes
import subprocess
from fastapi import UploadFile, File
from io import BytesIO

# check if given file path corresponds to a supported image MIME type.
def is_supported_image(file_path):
    supported_image_mimetypes = {
        "image/png", "image/jpeg", "image/webp", "image/heic", "image/heif"
    }
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type in supported_image_mimetypes

# check if given file path corresponds to a supported audio MIME type.
def is_supported_audio(file_path):
    supported_audio_mimetypes = {
        "audio/wav", "audio/mp3", "audio/aiff", "audio/aac",
        "audio/ogg", "audio/flac"
    }
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type in supported_audio_mimetypes

# check if file path corresponds to a supported video MIME type.
def is_supported_video(file_path):
    supported_video_mimetypes = {
        "video/mp4", "video/mpeg", "video/mov", "video/avi",
        "video/x-flv", "video/mpg", "video/webm", "video/wmv",
        "video/3gpp"
    }
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type in supported_video_mimetypes

def write_fastapi_file(file: UploadFile):
    file_path = f"./uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(file.read())
    return file_path

def read_file(file_path):
    try:

        # read audio
        if file_path.endswith(".mp3") or file_path.endswith(".wav"):
            file = {
                "mime_type": "audio/mp3",
                "data": pathlib.Path(file_path).read_bytes()
            }
        
        # read image
        elif (file_path.endswith(".png") or file_path.endswith(".jpg") or file_path.endswith(".webp")):
            file = PIL.Image.open(file_path)

        # Handle word file
        elif file_path.endswith(".docx"):
            doc = docx.Document(file_path)
            file = ""
            for para in doc.paragraphs:
                file += para.text # + "\n"

        # Handle webpage
        elif file_path.startswith("http://") or file_path.startswith("https://"):
            response = requests.get(file_path)
            soup = BeautifulSoup(response.content, 'html.parser')
            file = soup.get_text(separator='\n')

        # Handle plain text files
        elif file_path.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as file:
                file = file.read()

        # read doc/video
        else:
            try:
                file = upload_file(file_path)
            except:
                raise ValueError("Couldn't read the given file: {}".format(file_path))

    except Exception as e:
            raise ValueError("Error whil reding the file: {}".format(file_path))

    return file

def upload_file(path):
    # upload to google
    video_file = genai.upload_file(path=path)
    while video_file.state.name == "PROCESSING":
        time.sleep(1)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError(video_file.state.name)
    return video_file

def read_as_bytesio(file: UploadFile):
    file_like = file.file
    byte_stream = BytesIO(file_like.read())
    byte_stream.seek(0)
    return byte_stream

def convert_video_to_audio(video_path, audio_format="mp3"):

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"The video file '{video_path}' does not exist.")

    # audio path
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_audio_path = os.path.join(os.path.dirname(video_path), f"{base_name}.{audio_format}")

    # FFmpeg command to extract audio
    command = [
        "ffmpeg",
        "-i", video_path,        # Input video file
        "-vn",                   # Skip the video stream
        "-acodec", "copy",       # Copy the audio codec directly (or use a different codec, e.g., "libmp3lame")
        output_audio_path        # Output audio file
    ]

    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_audio_path

def get_current_device():
    """
    Returns the name of the available device
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    elif is_torch_npu_available():
        return "npu"
    else:
        return "cpu"