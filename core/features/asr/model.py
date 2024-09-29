import subprocess
import json
from nemo.collections.asr.models import EncDecMultiTaskModel
import moviepy.editor as mp
from pydub import AudioSegment
import os

# load model
canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b').to("cuda:0")

# update dcode params
decode_cfg = canary_model.cfg.decoding
decode_cfg.beam.beam_size = 1
canary_model.change_decoding_strategy(decode_cfg)
canary_model.to(canary_model.device)