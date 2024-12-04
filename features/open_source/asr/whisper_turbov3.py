import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from torch.nn.attention import SDPBackend, sdpa_kernel
from datasets import load_dataset, Audio

class WhisperTranscriber(BaseTranscriber):

    def __init__(self, model_id="openai/whisper-large-v3-turbo", attn="sdpa", compile_forward=False):
        self.compile_forward = compile_forward

        torch.set_float32_matmul_precision("high")

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, attn_implementation=attn
        )
        model.to(device)

        if self.compile_forward:
            # Enable static cache and compile the forward pass
            model.generation_config.cache_implementation = "static"
            model.generation_config.max_new_tokens = 256
            model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

        processor = AutoProcessor.from_pretrained(model_id, chunk_length=33)

        self.default_gen_kwargs = {
                "language": "en",
                "max_new_tokens": 256,
                "num_beams": 1,
                "condition_on_prev_tokens": False,
                "compression_ratio_threshold": 1.35,  # zlib compression ratio threshold (in token space)
                "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "return_timestamps": False,
            }

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=10,
            batch_size=16,
            torch_dtype=torch_dtype,
            device=device,
        )

        if self.compile_forward:

            dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
            sample = dataset[0]["audio"]

            # 2 warmup steps
            print("Warming up...")
            for _ in range(2):
                with sdpa_kernel(SDPBackend.MATH):
                    _ = self.pipe(sample.copy(), generate_kwargs={"min_new_tokens": 256, "max_new_tokens": 256})
            print("Done...")

    def transcribe_audio(self, audio_path, need_timestamp=True, **kwargs):
        audio_path = [audio_path] if isinstance(audio_path, str) else audio_path

        # generation kwargs
        all_kwargs = {**self.default_gen_kwargs, **kwargs}

        # compiled model run (4.5x expected speedup)
        if self.compile_forward:
            with sdpa_kernel(SDPBackend.MATH):
                # ASR pipeline passthrough
                result = self.pipe(audio_path, batch_size=len(audio_path), generate_kwargs=all_kwargs, return_timestamps=need_timestamp) # timestamp true for sentence level and "word" word level
        else:
            with torch.no_grad():
                # ASR pipeline passthrough
                result = self.pipe(audio_path, batch_size=len(audio_path), generate_kwargs=all_kwargs, return_timestamps=need_timestamp)

        return result[0]["text"] #if not need_timestamp else result["chunks"]
    

transcriber = WhisperTranscriber()

def transcribe_audio(audio_path):
    return transcriber.transcribe_audio(audio_path)