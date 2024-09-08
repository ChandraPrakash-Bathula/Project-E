import subprocess
import json
from nemo.collections.asr.models import EncDecMultiTaskModel

# load model
canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')

# update dcode params
decode_cfg = canary_model.cfg.decoding
decode_cfg.beam.beam_size = 1
canary_model.change_decoding_strategy(decode_cfg)


def transcribe_audio(audio_path):

    with open("manifest.json") as input_json:
        data = json.load(input_json)
        data["audio_filepath"] = audio_path

    transcript = canary_model.transcribe(
            "manisfest.json",
            batch_size=16,  # batch size to run the inference with
        )


    return transcript

def convert_video_to_audio(video_file):
    audio_path = video_file.filename.rsplit('.', 1)[0] + '.wav'
    cmd = f'ffmpeg -i "{video_file.filename}" -ab 160k -ar 44100 -vn "{audio_path}"'
    subprocess.call(cmd, shell=True)
    return audio_path
