from nemo.collections.asr.models import EncDecMultiTaskModel
from features.utils import get_current_device

# load model
canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')

# update dcode params
decode_cfg = canary_model.cfg.decoding
decode_cfg.beam.beam_size = 1
canary_model.change_decoding_strategy(decode_cfg)
canary_model.to(get_current_device())

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