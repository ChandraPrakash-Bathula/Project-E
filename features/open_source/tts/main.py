import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, AutoFeatureExtractor
import soundfile as sf
import scipy
from features.utils import get_current_device

repo_id = "parler-tts/parler-tts-mini-v1"
attn_implementation = "sdpa" # "sdpa" 1.4x or "flash_attention_2"
device=get_current_device()

model = ParlerTTSForConditionalGeneration.from_pretrained(
            repo_id,
            attn_implementation=attn_implementation,
            cache_dir="./model/"
        ).to(device)
tokenizer = AutoTokenizer.from_pretrained(repo_id, cache_dir="./model/")
feature_extractor = AutoFeatureExtractor.from_pretrained(repo_id, cache_dir="./model/")

description = "Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise."
input_ids = tokenizer(description, return_tensors="pt").to(model.device)

def text2speech(text: str, file_path):
    prompt_input_ids = tokenizer(text, return_tensors="pt").to(model.device)

    generation = model.generate(
        input_ids=input_ids.input_ids,
        attention_mask=input_ids.attention_mask,
        prompt_input_ids=prompt_input_ids.input_ids,
        prompt_attention_mask=prompt_input_ids.attention_mask,
        do_sample=True,
        return_dict_in_generate=True
    )
    audio_data = generation.sequences[0, :generation.audios_length[0]]

    audio_arr = audio_data.cpu().numpy().squeeze()
    scipy.io.wavfile.write(file_path, rate=feature_extractor.sampling_rate, data=audio_arr)