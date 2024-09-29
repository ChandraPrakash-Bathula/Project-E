from .model import canary_model

def transcribe_audio(audio_path):

    transcript = canary_model.transcribe(
        [audio_path],
        batch_size=4,
        # task='asr',
        # source_lang='en',
        # target_lang='en',
        # pnc=True
    )
    return transcript
