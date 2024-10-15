from whisper_turbov3 import transcribe_audio
import moviepy.editor as mp
from pydub import AudioSegment
import os

def convert_video_to_audio(video_path):
    audio_path = video_path.rsplit(".", 1)[0] + ".wav"
    video = mp.VideoFileClip(video_path)

    temp_path = "extracted_audio.wav"
    video.audio.write_audiofile(temp_path)

    sound = AudioSegment.from_wav(temp_path)
    sound = sound.set_channels(1)

    sound.export(audio_path, format="wav")
    os.remove(temp_path)
    return audio_path

def transcribe_video(video_path):
    audio_path = convert_video_to_audio(video_path)
    transcript = transcribe_audio(audio_path)
    return transcript