from transformers import pipeline
import torch
from features.utils import extract_content_from_file
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, AutoConfig, GenerationConfig
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
import json

class SummarizationOutput(BaseModel):
    summary: str = Field(description="Summary of the content")
character_level_parser = JsonSchemaParser(SummarizationOutput.model_json_schema())

# Initialize the configs
repo_id = "microsoft/Phi-3.5-mini-instruct"
cache_dir="./model/"
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_compute_dtype=getattr(torch, "float16"),
    bnb_4bit_quant_type="fp4",
    bnb_4bit_use_double_quant=False
)
config = AutoConfig.from_pretrained(repo_id, cache_dir=cache_dir)

# Init the tokenizer
tokenizer = AutoTokenizer.from_pretrained(repo_id)

# Init generation config and model
prefix_allowed_tokens_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, character_level_parser)

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=repo_id,
    torch_dtype="auto",
    # trust_remote_code=True,
    quantization_config=bnb_config,
    config=config,
    device_map="auto",
)
model.eval()

lm_pipline = pipeline("text-generation", model=model, tokenizer=tokenizer, config=config, framework="pt", device_map="auto", prefix_allowed_tokens_fn=prefix_allowed_tokens_fn)

generate_kwargs = dict(
        max_new_tokens=1024,
        do_sample=True,
        output_scores=True,
        return_dict_in_generate=False,
        return_full_text=False,
)

def summarize_with_slm(file_path):
    text = extract_content_from_file(file_path)
    message = [
        {"role": "system", "content": f"You are a Intelligent summarizer, who give a concise summary of the given content by carefully reading throughout them. You MUST answer using the following json schema: {SummarizationOutput.model_json_schema()}"},
        {"role": "user", "content": f"Content: \n\n{text}"}
    ]

    output = lm_pipline(message, **generate_kwargs)
    json_response = output[0]["generated_text"].strip()
    kv = json.loads(json_response)
    return kv["summary"]