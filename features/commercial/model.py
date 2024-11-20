import os
import google.generativeai as genai

genai.configure(api_key=os.environ["API_KEY"])

sys_instruction = """
You are a super intelligent & friendly assistant, Help user with their queries.
"""

gen_config = genai.GenerationConfig(
    candidate_count=1,
    temperature=1.1,
    top_k=65,
    top_p=0.95,
    max_output_tokens=4096
)

genai_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-001",
    generation_config=gen_config,
    system_instruction=sys_instruction
)