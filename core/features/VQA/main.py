import sys
from features.VQA import processor, model, yolo_model
from features.VQA.config import translate_text, detect_objects, show_image_with_boxes

def chat_with_image(image_path, text, target_language="en"):
    try:
        from PIL import Image
        import torch
        from transformers import TextIteratorStreamer, GenerationConfig
        from threading import Thread

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        image = Image.open(image_path).convert("RGB")
        boxes = detect_objects(image, yolo_model)
        show_image_with_boxes(image, boxes)

        text_translated = translate_text(text, target_language)
        inputs = processor.process(images=[image], text=text_translated)
        inputs = {k: v.to(device).unsqueeze(0) for k, v in inputs.items()}

        streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            generation_config=GenerationConfig(max_new_tokens=200, temperature=0.7, top_k=50, stop_strings=["<|endoftext|>"]),
            streamer=streamer,
            tokenizer=processor.tokenizer
        )

        thread = Thread(target=model.generate_from_batch, args=(inputs,), kwargs=generation_kwargs)
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            thread.start()

        buffer = ""
        for new_text in streamer:
            buffer += new_text
        return buffer
    except Exception as e:
        yield f"An error occurred during processing: {str(e)}"